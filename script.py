"""
Генератор Telegram-ботов на основе LangChain.

Цепочка из 3 звеньев:
  1. analysis_chain — анализ задания, выделение команд, обработчиков, зависимостей
  2. code_chain    — генерация полного кода бота (aiogram 3.x)
  3. review_chain  — проверка кода на корректность

Использование:
  python script.py "Бот, который отправляет случайные мемы"
"""

import sys
import os
import ast
import time
import logging
import textwrap
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env (если файл существует)
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ──────────────────────────────────────────────
# Логирование
# ──────────────────────────────────────────────
logger = logging.getLogger("botgen")


class ColorFormatter(logging.Formatter):
    """Форматтер с цветами и эмодзи для консоли."""

    COLORS = {
        logging.DEBUG:    "\033[90m",     # серый
        logging.INFO:     "\033[36m",     # голубой
        logging.WARNING:  "\033[33m",     # жёлтый
        logging.ERROR:    "\033[31m",     # красный
        logging.CRITICAL: "\033[1;31m",   # жирный красный
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


def setup_logging(verbose: bool = False) -> None:
    """Настройка логирования: консоль + опциональный файл."""
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    # Консольный хэндлер с цветами
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = ColorFormatter("%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Файловый хэндлер (всегда DEBUG)
    log_file = Path("generation.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.debug("Логирование инициализировано (verbose=%s, log_file=%s)", verbose, log_file.resolve())


# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """Инициализация LLM. Поддерживает любую OpenAI-совместимую модель."""
    model_name = os.getenv("MODEL_NAME", "gpt-5.4-mini-2026-03-17")
    logger.debug("Инициализация LLM: model=%s, temperature=%.2f", model_name, temperature)
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
    )


# ──────────────────────────────────────────────
# 1. ANALYSIS CHAIN
# ──────────────────────────────────────────────
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        textwrap.dedent("""\
            Ты — архитектор Telegram-ботов.
            Пользователь даст текстовое описание бота.
            Проанализируй задание и верни ТОЛЬКО структурированный ответ
            в следующем формате (без маркдауна, без обёрток в ```):

            COMMANDS:
            - /command1 — описание
            - /command2 — описание

            HANDLERS:
            - тип_обработчика: описание (например: message_handler: обработка текстовых сообщений)

            RESPONSE_FORMAT:
            - описание формата ответов бота (текст, фото, инлайн-кнопки и т.д.)

            DATABASE:
            - нужна / не нужна (и почему)

            DEPENDENCIES:
            - список дополнительных python-пакетов (кроме aiogram), если нужны
        """),
    ),
    ("human", "{task_description}"),
])


def build_analysis_chain():
    return ANALYSIS_PROMPT | get_llm(temperature=0.1) | StrOutputParser()


# ──────────────────────────────────────────────
# 2. CODE GENERATION CHAIN
# ──────────────────────────────────────────────
CODE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        textwrap.dedent("""\
            Ты — senior Python-разработчик, специализирующийся на Telegram-ботах.
            На вход получишь:
              • Исходное описание бота
              • Результат анализа (команды, обработчики, зависимости)

            Сгенерируй ПОЛНЫЙ, РАБОЧИЙ код Telegram-бота, соблюдая требования:

            1. Используй aiogram 3.x ПОСЛЕДНЕЙ ВЕРСИИ (3.7+).
               - Импорты: from aiogram import Bot, Dispatcher, Router, F
               - from aiogram.filters import CommandStart, Command
               - from aiogram.types import Message
               - from aiogram.client.default import DefaultBotProperties
               - from aiogram.enums import ParseMode
               - Router() для регистрации хэндлеров
               - dp.include_router(router)
               - await dp.start_polling(bot)
               - ВАЖНО: Bot(token=..., default=DefaultBotProperties(parse_mode=ParseMode.HTML))
               - НЕЛЬЗЯ использовать Bot(parse_mode=...) — это удалено в 3.7.0!
               - Все типы aiogram — pydantic-модели, ТОЛЬКО keyword аргументы:
                 KeyboardButton(text="/meme")  — ПРАВИЛЬНО
                 KeyboardButton("/meme")       — НЕПРАВИЛЬНО!
                 InlineKeyboardButton(text="Мем", callback_data="meme")  — ПРАВИЛЬНО
               - ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="...")]]) — ПРАВИЛЬНО
               - НЕ используй .add() или .row() — их НЕТ в aiogram 3.x!
            2. Все хэндлеры — async def.
            3. Используй python-dotenv для загрузки переменных:
               from dotenv import load_dotenv
               load_dotenv()
               BOT_TOKEN = os.getenv("BOT_TOKEN")
            4. Никаких заглушек и TODO — код должен запускаться as-is.
            5. Добавь необходимые импорты (os, asyncio, logging и т.д.).
            6. В конце файла — блок if __name__ == "__main__" с asyncio.run(main()).
            7. Добавляй краткие комментарии к ключевым блокам.
            8. НЕ используй локальные файлы и папки (например, папку "memes/"),
               ЕСЛИ пользователь явно не указал это в описании бота.
               Если нужны данные (мемы, картинки, цитаты и т.д.) — используй:
               - Встроенный список URL/данных прямо в коде (list/dict)
               - Бесплатные публичные HTTP API (например: meme-api.com, random.dog и т.д.)
               - Генерация данных средствами Python (random, string и т.д.)
               Код должен работать сразу после запуска без дополнительной настройки.

            Верни ТОЛЬКО Python-код, без маркдауна, без обёрток в ```.
        """),
    ),
    (
        "human",
        "Описание бота:\n{task_description}\n\nАнализ:\n{analysis}",
    ),
])


def build_code_chain():
    return CODE_PROMPT | get_llm(temperature=0.2) | StrOutputParser()


# ──────────────────────────────────────────────
# 3. REVIEW CHAIN
# ──────────────────────────────────────────────
REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        textwrap.dedent("""\
            Ты — code reviewer для Python и aiogram 3.x (версия 3.7+).
            Тебе дан код Telegram-бота. Проверь его:

            1. Синтаксические ошибки — код должен парситься ast.parse().
            2. Корректные импорты — все используемые модули импортированы.
            3. Структура запуска — есть if __name__ == "__main__", asyncio.run(main()),
               Dispatcher, Bot, Router подключены правильно.
            4. Все хэндлеры используют async def.
            5. BOT_TOKEN читается из os.getenv("BOT_TOKEN").
            6. Нет заглушек, TODO, pass-only функций.

            ВАЖНЫЕ ПРАВИЛА:
            - НЕ МЕНЯЙ импорты, если они корректны.
            - В aiogram 3.7+ правильный способ указания parse_mode:
              from aiogram.client.default import DefaultBotProperties
              Bot(token=..., default=DefaultBotProperties(parse_mode=ParseMode.HTML))
              Это ПРАВИЛЬНЫЙ код — НЕ МЕНЯЙ его!
            - Будь КОНСЕРВАТИВЕН: если не уверен что что-то сломано — НЕ ТРОГАЙ.
            - Исправляй только ЯВНЫЕ ошибки.

            Если код корректен — верни его БЕЗ ИЗМЕНЕНИЙ.
            Если есть ошибки — ИСПРАВЬ только их и верни исправленный код.

            Верни ТОЛЬКО Python-код, без пояснений, без маркдауна, без обёрток в ```.
        """),
    ),
    ("human", "{code}"),
])


def build_review_chain():
    return REVIEW_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


# ──────────────────────────────────────────────
# 4. FIX CHAIN (исправление конкретных ошибок)
# ──────────────────────────────────────────────
FIX_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        textwrap.dedent("""\
            Ты — Python-разработчик. Тебе дан код Telegram-бота с конкретной ошибкой.
            
            ПРАВИЛА:
            - Исправь ТОЛЬКО указанную ошибку.
            - НЕ меняй и НЕ удаляй остальной код.
            - НЕ меняй рабочие импорты. В частности, import
              from aiogram.client.default import DefaultBotProperties — ПРАВИЛЬНЫЙ.
            - Верни ТОЛЬКО исправленный Python-код, без пояснений, без маркдауна, без обёрток в ```.
        """),
    ),
    ("human", "Ошибка: {error}\n\nКод:\n{code}"),
])


def build_fix_chain():
    return FIX_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


# ──────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────
def extract_code(text: str) -> str:
    """
    Извлекает Python-код из ответа LLM.
    Обрабатывает случаи:
      - Чистый код без обёрток
      - Код в ```python ... ```
      - Пояснительный текст + код в ``` блоке
    """
    import re

    text = text.strip()

    # Ищем все блоки ```python ... ``` или ``` ... ```
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Берём самый длинный блок кода
        code = max(matches, key=len).strip()
        logger.debug("Извлечён код из markdown-блока (%d символов)", len(code))
        return code

    # Если нет markdown-блоков — ищем начало кода по маркерам
    # Берём САМЫЙ РАННИЙ маркер, чтобы не обрезать import перед from
    markers = ("import ", "from ", "# -*-", "#!/")
    first_pos = len(text)
    first_marker = None
    for marker in markers:
        idx = text.find(marker)
        if idx >= 0 and idx < first_pos:
            first_pos = idx
            first_marker = marker

    if first_marker and first_pos > 0:
        code = text[first_pos:].strip()
        logger.debug("Извлечён код по маркеру '%s' (отброшено %d символов пояснений)",
                    first_marker, first_pos)
        return code

    # Ничего не нашли или код начинается с маркера — возвращаем как есть
    return text


def validate_code(code: str) -> tuple[bool, str]:
    """
    Комплексная проверка кода.
    Возвращает (is_valid, error_description).
    """
    # 1. Синтаксис
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        error = f"Синтаксическая ошибка: {e.msg} (строка {e.lineno})"
        logger.warning("⚠️  %s", error)
        return False, error

    # 2. Собираем все импортированные имена
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)

    # Проверяем стандартные модули, которые часто забывают импортировать
    STDLIB_NAMES = {"os", "asyncio", "logging", "random", "json", "sys", "re"}
    used_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in STDLIB_NAMES:
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in STDLIB_NAMES:
                used_names.add(node.value.id)

    missing = used_names - imported_names
    if missing:
        error = f"Отсутствуют импорты: {', '.join(sorted(missing))}"
        logger.warning("⚠️  %s", error)
        return False, error

    return True, ""


# ──────────────────────────────────────────────
# Основной пайплайн
# ──────────────────────────────────────────────
def run_pipeline(task_description: str) -> str:
    """
    Запускает полную цепочку:
      analysis → code_generation → review
    Возвращает финальный Python-код.
    """
    pipeline_start = time.perf_counter()

    logger.info("═" * 55)
    logger.info("🔗 Запуск цепочки генерации Telegram-бота")
    logger.info("═" * 55)
    logger.info("📝 Задание: %s", task_description)
    logger.debug("Длина описания: %d символов", len(task_description))

    # ── Звено 1: Анализ ──
    logger.info("🔍 [1/3] Анализ задания...")
    t0 = time.perf_counter()
    analysis_chain = build_analysis_chain()
    analysis = analysis_chain.invoke({"task_description": task_description})
    elapsed = time.perf_counter() - t0
    logger.info("   Анализ завершён за %.1f сек", elapsed)
    logger.debug("Результат анализа (%d символов):\n%s", len(analysis), analysis)

    # ── Звено 2: Генерация кода ──
    logger.info("⚙️  [2/3] Генерация кода бота...")
    t0 = time.perf_counter()
    code_chain = build_code_chain()
    raw_code = code_chain.invoke({
        "task_description": task_description,
        "analysis": analysis,
    })
    raw_code = extract_code(raw_code)
    elapsed = time.perf_counter() - t0
    lines_count = len(raw_code.splitlines())
    logger.info("   Код сгенерирован за %.1f сек (%d строк, %d символов)",
                elapsed, lines_count, len(raw_code))
    logger.debug("Сгенерированный код:\n%s", raw_code[:500] + "..." if len(raw_code) > 500 else raw_code)

    # ── Звено 3: Ревью + повторные попытки при ошибках ──
    MAX_RETRIES = 3
    review_chain = build_review_chain()
    current_code = raw_code
    is_valid, error_msg = validate_code(raw_code)
    pre_review_valid = is_valid

    if pre_review_valid:
        logger.debug("Код до ревью корректен")

    fix_chain = build_fix_chain()

    for attempt in range(1, MAX_RETRIES + 1):
        if attempt == 1:
            # Первый проход — стандартное ревью
            logger.info("🔎 [3/3] Проверка и ревью кода...")
            t0 = time.perf_counter()
            reviewed_code = review_chain.invoke({"code": current_code})
        else:
            # Повторные проходы — целенаправленное исправление ошибки
            logger.info("🔧 [3/3] Исправление ошибки (попытка %d/%d): %s",
                       attempt, MAX_RETRIES, error_msg)
            t0 = time.perf_counter()
            reviewed_code = fix_chain.invoke({"code": current_code, "error": error_msg})

        reviewed_code = extract_code(reviewed_code)
        elapsed = time.perf_counter() - t0

        # Сравнение: были ли изменения
        if reviewed_code.strip() == current_code.strip():
            logger.info("   Завершено за %.1f сек — код не изменён ✓", elapsed)
        else:
            diff_lines = abs(len(reviewed_code.splitlines()) - lines_count)
            logger.info("   Завершено за %.1f сек — код ИЗМЕНЁН (±%d строк)",
                        elapsed, diff_lines)
            logger.debug("Код после обработки:\n%s",
                         reviewed_code[:500] + "..." if len(reviewed_code) > 500 else reviewed_code)

        # Проверка кода
        is_valid, error_msg = validate_code(reviewed_code)
        if is_valid:
            logger.info("✅ Код корректен.")
            current_code = reviewed_code
            break
        else:
            # Если ревью сломало рабочий код — откат
            if pre_review_valid and attempt == 1:
                logger.warning("⚠️  Ревью СЛОМАЛО рабочий код — откат к оригиналу.")
                current_code = raw_code
                break
            if attempt < MAX_RETRIES:
                logger.warning("🔄 Ошибка: %s — отправляю на исправление...", error_msg)
                current_code = reviewed_code
            else:
                logger.error("❌ Не удалось исправить за %d попыток.", MAX_RETRIES)
                current_code = reviewed_code

    reviewed_code = current_code

    total_elapsed = time.perf_counter() - pipeline_start
    logger.info("⏱  Общее время генерации: %.1f сек", total_elapsed)

    return reviewed_code


def main():
    # Флаг --verbose / -v для расширенного вывода
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--verbose", "-v")]

    if not args:
        print("Использование: python script.py [-v|--verbose] \"Описание бота\"")
        print('Пример: python script.py "Бот, который отправляет случайные мемы"')
        sys.exit(1)

    setup_logging(verbose=verbose)

    task_description = args[0]
    logger.info("Модель: %s", os.getenv("MODEL_NAME", "gpt-4o"))

    final_code = run_pipeline(task_description)

    # Сохраняем результат
    output_path = Path("generated_bot.py")
    output_path.write_text(final_code, encoding="utf-8")

    logger.info("═" * 55)
    logger.info("📦 Бот сохранён в: %s", output_path.resolve())
    logger.info("═" * 55)
    logger.info("Для запуска:")
    logger.info("  set BOT_TOKEN=ваш_токен")
    logger.info("  python %s", output_path)


if __name__ == "__main__":
    main()
