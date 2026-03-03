import argparse
import importlib
import os
import tempfile
from itertools import chain
from typing import List, Tuple, Union

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup, concatenate_poses
from spoken_to_signed.gloss_to_pose.coverage import CoverageStats, TokenCoverage
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str, **kwargs) -> List[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language, **kwargs)


def _make_lookup(lexicon: str) -> CSVPoseLookup:
    fingerspelling_lookup = FingerspellingPoseLookup()
    return CSVPoseLookup(lexicon, backup=fingerspelling_lookup)


def _gloss_to_pose(
    sentences: List[Gloss],
    pose_lookup: CSVPoseLookup,
    spoken_language: str,
    signed_language: str,
    coverage_info: bool = False,
) -> Union[Pose, Tuple[Pose, List[List[TokenCoverage]]]]:
    results = [
        gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language, coverage_info=coverage_info)
        for gloss in sentences
    ]

    if not coverage_info:
        poses = results
        return poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False)

    poses = [pose for pose, _ in results]
    all_token_coverages = [coverages for _, coverages in results]
    return (
        poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False),
        all_token_coverages,
    )


def _get_models_dir():
    home_dir = os.path.expanduser("~")
    sign_dir = os.path.join(home_dir, ".sign")
    os.makedirs(sign_dir, exist_ok=True)
    models_dir = os.path.join(sign_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _pose_to_video(pose: Pose, video_path: str):
    models_dir = _get_models_dir()
    pix2pix_path = os.path.join(models_dir, "pix2pix.h5")
    if not os.path.exists(pix2pix_path):
        print("Downloading pix2pix model")
        import urllib.request
        urllib.request.urlretrieve(
            "https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media",
            pix2pix_path)

    import subprocess

    try:
        subprocess.run(["command", "-v", "pose_to_video"], shell=True, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "The command 'pose_to_video' does not exist. Please install the `transcription` package using "
            "`pip install git+https://github.com/sign-language-processing/transcription`")

    pose_path = tempfile.mktemp(suffix=".pose")
    with open(pose_path, "wb") as f:
        pose.write(f)

    args = ["pose_to_video", "--type=pix_to_pix",
            "--model", pix2pix_path,
            "--pose", pose_path,
            "--video", video_path,
            "--upscale"]
    print(" ".join(args))
    subprocess.run(args, shell=True, check=True)


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=['simple', 'spacylemma', 'rules', 'nmt'], required=True)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lexicon", type=str)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.lexicon:
        lookup = CSVPoseLookup(pre_args.lexicon)
        spoken_languages = list(lookup.words_index.keys())
        signed_languages = set(chain.from_iterable(lookup.words_index[lang].keys() for lang in spoken_languages))
    else:
        spoken_languages = ['de', 'fr', 'it', 'en']
        signed_languages = ['sgg', 'gsg', 'bfi', 'ase']

    parser.add_argument("--spoken-language", choices=spoken_languages, required=True)
    parser.add_argument("--signed-language", choices=signed_languages, required=True)


def _print_token_coverage(all_token_coverages: List[List[TokenCoverage]]):
    total = sum(len(s) for s in all_token_coverages)
    matched = sum(tc.matched for s in all_token_coverages for tc in s)
    for sentence_coverages in all_token_coverages:
        for tc in sentence_coverages:
            status = "✓" if tc.matched else "✗"
            print(f"  {status}  {tc.word} -> {tc.gloss}")
    print(f"Coverage: {matched / total:.3f} ({matched}/{total} tokens matched)")


def text_to_gloss():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args = args_parser.parse_args()

    print("Text to gloss")
    print("Input text:", args.text)
    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    print("Output gloss:", sentences)


def pose_to_video():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    with open(args.pose, "rb") as f:
        pose = Pose.read(f.read())

    _pose_to_video(pose, args.video)

    print("Pose to video")
    print("Input pose:", args.pose)
    print("Output video:", args.video)


def text_to_gloss_to_pose():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--coverage-info", action="store_true",
                             help="Print per-token gloss coverage to stdout.")
    args_parser.add_argument("--coverage-stats", type=str, default=None,
                             help="Path to save per-token coverage statistics as a JSON file.")
    args_parser.add_argument("--pose", type=str, required=True)
    args = args_parser.parse_args()

    need_coverage = args.coverage_info or args.coverage_stats is not None

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose_lookup = _make_lookup(args.lexicon)
    result = _gloss_to_pose(sentences, pose_lookup, args.spoken_language, args.signed_language, need_coverage)

    print("Text to gloss to pose")
    print("Input text:", args.text)
    print("Output pose:", args.pose)

    if need_coverage:
        pose, all_token_coverages = result
        _print_token_coverage(all_token_coverages)
        if args.coverage_stats:
            stats = CoverageStats()
            for sentence_coverages in all_token_coverages:
                stats.add_sentence(sentence_coverages)
            stats.save(args.coverage_stats)
            print(f"Coverage stats saved to: {args.coverage_stats}")
    else:
        pose = result

    with open(args.pose, "wb") as f:
        pose.write(f)


def text_to_gloss_to_pose_bulk():
    args_parser = argparse.ArgumentParser(
        description="Translate a file of texts (one per line) into pose files in bulk.")
    args_parser.add_argument("--texts", type=str, required=True,
                             help="Path to a text file with one input sentence per line.")
    args_parser.add_argument("--glosser", choices=['simple', 'spacylemma', 'rules', 'nmt'], required=True)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--spoken-language", type=str, required=True)
    args_parser.add_argument("--signed-language", type=str, required=True)
    args_parser.add_argument("--output-dir", type=str, required=True,
                             help="Directory where output .pose files are written (named 000000.pose, 000001.pose, …).")
    args_parser.add_argument("--coverage-stats", type=str, default=None,
                             help="Path to save aggregated per-token coverage statistics as a JSON file.")
    args = args_parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.texts, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    need_coverage = args.coverage_stats is not None
    stats = CoverageStats() if need_coverage else None
    pose_lookup = _make_lookup(args.lexicon)

    for i, text in enumerate(texts):
        sentences = _text_to_gloss(text, args.spoken_language, args.glosser)
        result = _gloss_to_pose(sentences, pose_lookup, args.spoken_language, args.signed_language, need_coverage)

        if need_coverage:
            pose, all_token_coverages = result
            for sentence_coverages in all_token_coverages:
                stats.add_sentence(sentence_coverages)
        else:
            pose = result

        pose_path = os.path.join(args.output_dir, f"{i:06d}.pose")
        with open(pose_path, "wb") as f:
            pose.write(f)
        print(f"[{i + 1}/{len(texts)}] {pose_path}")

    if need_coverage:
        stats.save(args.coverage_stats)
        print(f"Coverage stats saved to: {args.coverage_stats}")
        print(f"Overall coverage: {stats.fraction:.3f} ({stats.matched_tokens}/{stats.total_tokens} tokens matched)")


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser, signed_language=args.signed_language)
    pose_lookup = _make_lookup(args.lexicon)
    pose = _gloss_to_pose(sentences, pose_lookup, args.spoken_language, args.signed_language)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)


if __name__ == "__main__":
    text_to_gloss_to_pose()
