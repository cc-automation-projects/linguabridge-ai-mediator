import subprocess

from app.core.logger import logger


class AudioConverter:
    @staticmethod
    def convert_wav_to_ogg_opus(wav_bytes: bytes, target_sr: int = 24000) -> bytes:
        try:
            command = [
                "ffmpeg",
                "-i", "pipe:0",
                "-c:a", "libopus",
                "-b:a", "16k",
                "-ar", str(target_sr),
                "-ac", "1",
                "-vn",
                "-f", "ogg",
                "pipe:1"
            ]

            process = subprocess.run(
                command,
                input=wav_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )

            logger.info("audio_conversion_successful", input_size=len(wav_bytes), output_size=len(process.stdout))
            return process.stdout

        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg_conversion_failed", stderr=e.stderr.decode()[:200], exc_info=True)
            raise
        except FileNotFoundError:
            logger.critical("ffmpeg_binary_not_found", msg="Убедитесь, что ffmpeg установлен в системе/Docker-образе")
            raise


audio_converter = AudioConverter()
