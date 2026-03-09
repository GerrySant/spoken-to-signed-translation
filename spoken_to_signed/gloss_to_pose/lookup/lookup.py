import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

from pose_format import Pose

from spoken_to_signed.gloss_to_pose.languages import LANGUAGE_BACKUP
from spoken_to_signed.gloss_to_pose.concatenate import concatenate_poses
from spoken_to_signed.gloss_to_pose.lookup.gloss_normalization_helpers import (
    get_progressive_gloss_normalizers,
    is_number_token,
    should_normalize_integer_token,
    split_decimal,
)
from spoken_to_signed.gloss_to_pose.lookup.lru_cache import LRUCache
from spoken_to_signed.text_to_gloss.types import Gloss


class LookupResult(NamedTuple):
    pose: Optional[Pose]
    coverage_type: Optional[str]
    sub_elements: Optional[list]


@dataclass
class PairResult:
    word: str
    gloss: str
    pose: Optional[Pose] = field(default=None)
    coverage_type: Optional[str] = field(default=None)
    sub_elements: Optional[list] = field(default=None)


class PoseLookup:
    def __init__(self, rows: list, directory: str = None, backup: "PoseLookup" = None, cache: LRUCache = None):
        self.directory = directory

        self.words_index = self.make_dictionary_index(rows, based_on="words")
        self.glosses_index = self.make_dictionary_index(rows, based_on="glosses")

        self.backup = backup

        self.file_systems = {}
        self.cache = cache if cache is not None else LRUCache()

    def make_dictionary_index(self, rows: list, based_on: str):
        # As an attempt to make the index more compact in memory, we store a dictionary with only what we need
        languages_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in rows:
            term = d[based_on]
            lower_term = term.lower()
            languages_dict[d["spoken_language"]][d["signed_language"]][lower_term].append(
                {
                    "path": d["path"],
                    "term": term,
                    "start": int(d["start"]),
                    "end": int(d["end"]),
                    "priority": int(d["priority"]),
                }
            )
        return languages_dict

    def read_pose(self, pose_path: str):
        if pose_path.startswith("gs://"):
            if "gcs" not in self.file_systems:
                import gcsfs

                self.file_systems["gcs"] = gcsfs.GCSFileSystem(anon=True)

            with self.file_systems["gcs"].open(pose_path, "rb") as f:
                return Pose.read(f.read())

        if pose_path.startswith("https://"):
            raise NotImplementedError("Can't access pose files from https endpoint")

        if self.directory is None:
            raise ValueError("Can't access pose files without specifying a directory")

        pose_path = os.path.join(self.directory, pose_path)
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def get_pose(self, row):
        # Manage pose cache
        cached_pose = self.cache.get(row["path"])
        if cached_pose is None:
            pose = self.read_pose(row["path"])
            self.cache.set(row["path"], pose)
        pose = self.cache.get(row["path"])

        frame_time = 1000 / pose.body.fps
        start_frame = math.floor(row["start"] // frame_time)
        end_frame = math.ceil(row["end"] // frame_time) if row["end"] > 0 else -1
        return Pose(pose.header, pose.body[start_frame:end_frame])

    def get_best_row(self, rows, term: str):
        # Sort by priority: lower is "better"
        rows = sorted(rows, key=lambda x: x["priority"])
        # String match exact term
        for row in rows:
            if term == row["term"]:
                return row
        # Return the highest priority row
        return rows[0]

    def lookup(
        self,
        word: str,
        gloss: str,
        spoken_language: str,
        signed_language: str,
        source: str = None,
        number_placeholder: Optional[str] = None,
    ) -> LookupResult:
        preprocess_steps = get_progressive_gloss_normalizers()
        current_gloss = gloss

        # Attempts within the main language with progressive normalization
        for step_fn in preprocess_steps:
            if step_fn is not None:
                current_gloss = step_fn(current_gloss)

            lookup_list = [
                (self.words_index, (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, current_gloss)),
            ]

            for dict_index, (spoken_language, signed_language, term) in lookup_list:
                if spoken_language in dict_index:
                    if signed_language in dict_index[spoken_language]:
                        lower_term = term.lower()
                        if lower_term in dict_index[spoken_language][signed_language]:
                            rows = dict_index[spoken_language][signed_language][lower_term]
                            return LookupResult(self.get_pose(self.get_best_row(rows, term)), "lexicon", None)

        # For the backups: normalize the word only if the gloss represents a standalone integer.
        # This avoids altering decimals or alphanumeric tokens (e.g. "3.14", "A3").
        if should_normalize_integer_token(gloss):
            word = preprocess_steps[-2](word)

        # Backup strategy: decompose decimal numbers (e.g. "3.14" → "3" + "." + "14")
        # Each part is looked up independently — parts not found fall back to number_placeholder if provided.
        # sub_elements stores [part, found] pairs so callers can distinguish matched from unmatched parts.
        decimal_parts = split_decimal(gloss)
        if decimal_parts is not None:
            poses, parts_with_status = [], []
            for p in decimal_parts:
                try:
                    poses.append(self.lookup(p, p, spoken_language, signed_language, source).pose)
                    parts_with_status.append([p, True])
                except FileNotFoundError:
                    if number_placeholder is not None:
                        try:
                            poses.append(
                                self.lookup(number_placeholder, number_placeholder, spoken_language, signed_language, source).pose
                            )
                            parts_with_status.append([p, "placeholder"])
                        except FileNotFoundError:
                            parts_with_status.append([p, False])
                    else:
                        parts_with_status.append([p, False])
            if poses:
                return LookupResult(concatenate_poses(poses), "decimal_parts", parts_with_status)

        # Backup strategy: revert to backup sign language
        if signed_language in LANGUAGE_BACKUP:
            result = self.lookup(word, gloss, spoken_language, LANGUAGE_BACKUP[signed_language], source, number_placeholder)
            # If found in the backup language's own lexicon, label as language_backup; otherwise keep the type
            if result.coverage_type == "lexicon":
                return LookupResult(result.pose, "language_backup", None)
            return result

        # Backup strategy: revert to fingerspelling
        if self.backup is not None:
            return self.backup.lookup(word, gloss, spoken_language, signed_language, source, number_placeholder)

        # Final fallback for number tokens: use the placeholder gloss if provided
        if number_placeholder is not None and is_number_token(gloss):
            try:
                result = self.lookup(number_placeholder, number_placeholder, spoken_language, signed_language, source)
                return LookupResult(result.pose, "placeholder", None)
            except FileNotFoundError:
                pass

        raise FileNotFoundError

    def lookup_sequence(
        self,
        glosses: Gloss,
        spoken_language: str,
        signed_language: str,
        source: str = None,
        coverage_info: bool = False,
        number_placeholder: Optional[str] = None,
    ):
        def lookup_pair(pair):
            word, gloss = pair
            if word == "":
                return PairResult(word=word, gloss=gloss)

            try:
                result = self.lookup(word, gloss, spoken_language, signed_language, number_placeholder=number_placeholder)
                return PairResult(
                    word=word,
                    gloss=gloss,
                    pose=result.pose,
                    coverage_type=result.coverage_type,
                    sub_elements=result.sub_elements,
                )
            except FileNotFoundError as e:
                print(e)
                return PairResult(word=word, gloss=gloss)

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lookup_pair, glosses))

        poses = [r.pose for r in results if r.pose is not None]

        if len(poses) == 0:
            gloss_sequence = " ".join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        if coverage_info:
            from spoken_to_signed.gloss_to_pose.coverage import TokenCoverage

            token_coverages = [
                TokenCoverage(
                    word=r.word,
                    gloss=r.gloss,
                    matched=(r.coverage_type == "lexicon"),
                    coverage_type=r.coverage_type,
                    fingerspelled_keys=r.sub_elements,
                )
                for r in results
                if r.word != ""  # exclude empty/skipped tokens
            ]
            return poses, token_coverages

        return poses
