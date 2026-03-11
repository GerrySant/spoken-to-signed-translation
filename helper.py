import threading
_POSE_READ_LOCK = threading.Lock()


    def read_pose(self, pose_path: str):
        """Safely read a pose file, avoiding concurrent Pose.read issues."""
        if pose_path.startswith('gs://'):
            if 'gcs' not in self.file_systems:
                import gcsfs
                self.file_systems['gcs'] = gcsfs.GCSFileSystem(anon=True)
            with self.file_systems['gcs'].open(pose_path, "rb") as f:
                data = f.read()

        elif pose_path.startswith('https://'):
            raise NotImplementedError("Can't access pose files from https endpoint")

        else:
            if self.directory is None:
                raise ValueError("Can't access pose files without specifying a directory")
            full_path = os.path.join(self.directory, pose_path)
            with open(full_path, "rb") as f:
                data = f.read()

        # Serialize the actual Pose.read (thread-unsafe part)
        with _POSE_READ_LOCK:
            try:
                return Pose.read(data)
            except TypeError as e:
                # Add extra context for debugging corrupted/truncated files
                raise RuntimeError(
                    f"Failed to decode pose file {pose_path} "
                    f"(size={len(data)} bytes)"
                ) from e