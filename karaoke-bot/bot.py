import os
import shutil
import signal
import asyncio
import logging
import time
import subprocess
import boto3
from botocore.client import Config
from pathlib import Path
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from karaoke_utils import (
    prepare_audio, separate_audio, preprocess_audio,
    transcribe_audio, words_to_ass_karaoke, create_karaoke_video,
    cleanup_temp_files
)

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "karaoke")

MAX_AUDIO_SIZE_MB = 50
MAX_DURATION_SECONDS = 600  #10мин

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
logging.basicConfig(level=logging.INFO)

WORK_DIR = Path("/workspace/temp")
WORK_DIR.mkdir(exist_ok=True, parents=True)

def get_free_space_gb(path="/workspace"):
    stat = os.statvfs(path)
    return (stat.f_bavail * stat.f_frsize) / (1024**3)

async def shutdown():
    logging.info("Очистка перед завершением...")
    for item in WORK_DIR.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except:
            pass

def handle_signal():
    asyncio.create_task(shutdown())

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda s, f: handle_signal())

async def periodic_cleanup(interval_seconds=3600):
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            for item in WORK_DIR.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass
            logging.info("Периодическая очистка WORK_DIR выполнена")
        except Exception as e:
            logging.warning(f"Ошибка при очистке: {e}")

s3_client = boto3.client(
    's3',
    endpoint_url=f"http://{MINIO_ENDPOINT}",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

try:
    s3_client.create_bucket(Bucket=MINIO_BUCKET)
except:
    pass

def get_audio_duration(file_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", file_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(output.strip())
    except:
        return 0.0

def upload_to_minio(file_path: str, object_name: str, retries=3):
    for attempt in range(retries):
        try:
            extra_args = {
                'ContentDisposition': 'inline',
                'ContentType': 'video/mp4'
            }
            s3_client.upload_file(file_path, MINIO_BUCKET, object_name, ExtraArgs=extra_args)
            return f"https://mykaraokebot.win/{MINIO_BUCKET}/{object_name}?response-content-disposition=inline"
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "🎤 **Караоке-бот**\n\n"
        "Просто отправьте мне аудиофайл (mp3, wav, m4a) или голосовое сообщение.\n\n"
        f"📁 **Ограничения:**\n"
        f"• Размер файла до {MAX_AUDIO_SIZE_MB} МБ\n"
        f"• Длительность до {MAX_DURATION_SECONDS // 60} минут\n\n"
        "✅ Результат придёт видеофайлом или ссылкой на облачное хранилище.\n\n"
        "🎧 Начинайте прямо сейчас!"
    )

@dp.message(lambda message: message.audio or message.voice or message.video)
async def handle_media(message: types.Message):
    audio = message.audio or message.voice or message.video
    if not audio:
        await message.answer("Не удалось получить аудиофайл.")
        return

    file_size_mb = audio.file_size / (1024 * 1024)
    if file_size_mb > MAX_AUDIO_SIZE_MB:
        await message.answer(
            f"❌ Файл слишком большой ({file_size_mb:.1f} МБ). "
            f"Максимальный размер {MAX_AUDIO_SIZE_MB} МБ."
        )
        return

    if get_free_space_gb() < 2:
        await message.answer("❌ На сервере заканчивается место, попробуйте позже.")
        return

    processing_msg = await message.answer("⏳ Начинаю обработку... (3-5 минут)")
    start_time = time.time()

    audio_temp = NamedTemporaryFile(suffix=".input", delete=False, dir=WORK_DIR)
    await bot.download(audio, destination=audio_temp.name)
    audio_temp.close()

    wav_path = None
    vocal = None
    instr = None
    opt_audio = None
    ass_file = None
    video_path = None

    duration = get_audio_duration(audio_temp.name)
    if duration > MAX_DURATION_SECONDS:
        await message.answer(
            f"❌ Аудио слишком длинное ({duration // 60:.0f} мин {duration % 60:.0f} сек). "
            f"Максимум {MAX_DURATION_SECONDS // 60} минут."
        )
        cleanup_temp_files(audio_temp.name)
        await processing_msg.delete()
        return

    try:
        await processing_msg.edit_text("🎧 Конвертируем аудио...")
        wav_path = prepare_audio(audio_temp.name, output_audio=WORK_DIR / "input.wav")

        await processing_msg.edit_text("🔊 Разделяем голос и музыку...")
        vocal, instr = separate_audio(wav_path)

        opt_audio = preprocess_audio(wav_path, output_path=WORK_DIR / "opt.wav")

        await processing_msg.edit_text("📝 Распознаём текст...")
        words, full_text = transcribe_audio(opt_audio)
        if not words:
            await message.answer("❌ Не удалось распознать слова. Попробуйте другой файл.")
            return

        ass_file = words_to_ass_karaoke(words, output_ass=WORK_DIR / "karaoke.ass")

        await processing_msg.edit_text("🎬 Создаём видео...")
        video_path = create_karaoke_video(instr, ass_file, output_video=WORK_DIR / "result.mp4")

        await processing_msg.edit_text("☁️ Загружаем результат в облако...")
        unique_name = f"karaoke_{message.from_user.id}_{int(time.time())}.mp4"
        file_url = upload_to_minio(video_path, unique_name)

        elapsed = time.time() - start_time
        caption = f"✅ Готово за {elapsed:.1f} сек!\n🎥 Ваше видео: {file_url}"
        await message.answer(caption)

    except Exception as e:
        logging.exception("Ошибка")
        await message.answer(f"❌ Ошибка: {str(e)[:200]}")
    finally:
        cleanup_temp_files(audio_temp.name, wav_path, vocal, instr, opt_audio, ass_file, video_path)
        await processing_msg.delete()

@dp.message()
async def handle_other(message: types.Message):
    await message.answer(
        "🎤 **Чтобы создать караоке-видео, отправьте мне аудиофайл** (mp3, wav, m4a) или голосовое сообщение.\n\n"
        f"📁 **Ограничения:**\n"
        f"• Размер файла до {MAX_AUDIO_SIZE_MB} МБ\n"
        f"• Длительность до {MAX_DURATION_SECONDS // 60} минут\n\n"
        "Нажмите /start для повторного показа инструкции."
    )

async def main():
    asyncio.create_task(periodic_cleanup(3600))
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())

