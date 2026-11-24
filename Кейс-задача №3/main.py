"""Telegram FAQ bot for Synergy University.

Dependencies:
    pip install pyTelegramBotAPI spacy
    python -m spacy download ru_core_news_sm

Run the local intent checks:
    python main.py --self-test

Run the Telegram bot after setting TELEGRAM_BOT_TOKEN in the environment:
    python main.py
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

import spacy
import telebot


try:
    # The Russian pipeline provides tokenization and lemmatization.
    nlp = spacy.load("ru_core_news_sm", disable=["ner"])
except OSError as exc:
    raise SystemExit(
        "Не найдена русская модель spaCy. Установите её командой:\n"
        "python -m spacy download ru_core_news_sm"
    ) from exc


INTENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "greeting",
        ("привет", "здравствуйте", "добрый день", "доброе утро", "добрый вечер"),
    ),
    ("schedule", ("расписание", "занятие", "пара")),
    ("session", ("сессия", "экзамен", "зачёт", "зачет", "пересдача")),
    ("dean_office", ("деканат", "декан", "учебная часть")),
    ("scholarship", ("стипендия", "стипендиальный")),
    ("dormitory", ("общежитие", "заселение")),
    (
        "admission",
        ("поступление", "поступить", "поступать", "абитуриент", "приёмная комиссия", "приемная комиссия"),
    ),
    ("contacts", ("контакт", "телефон", "адрес", "почта")),
    ("thanks", ("спасибо", "благодарить")),
)

RESPONSES: dict[str, str] = {
    "start": (
        "Здравствуйте! Я бот-помощник Университета «Синергия». "
        "Могу подсказать, где искать сведения о расписании, сессии, "
        "стипендиях, общежитии и поступлении."
    ),
    "help": (
        "Спросите меня о расписании, сессии, деканате, стипендии, "
        "общежитии, поступлении или контактах Университета."
    ),
    "greeting": "Здравствуйте! Чем могу помочь?",
    "schedule": (
        "Расписание занятий доступно в личном кабинете и приложении Synergy LMS."
    ),
    "session": (
        "Актуальная информация о сессии, экзаменах и пересдачах публикуется "
        "в личном кабинете Университета."
    ),
    "dean_office": (
        "По вопросам деканата обратитесь в учебное подразделение своего факультета. "
        "Контакты факультетов доступны на официальном сайте Университета."
    ),
    "scholarship": (
        "Условия получения стипендии уточняйте в личном кабинете или "
        "в соответствующем подразделении Университета."
    ),
    "dormitory": (
        "Актуальную информацию о заселении и местах в общежитии уточняйте "
        "у Университета через официальные контакты."
    ),
    "admission": (
        "Информация о приёме и подаче документов размещена в разделе «Абитуриентам» "
        "на официальном сайте Университета."
    ),
    "contacts": "Официальные контакты Университета: https://synergy.ru/about/coordinat",
    "thanks": "Пожалуйста! Обращайтесь ещё.",
    "fallback": (
        "Извините, я не понял запрос. Попробуйте спросить о расписании, сессии, "
        "деканате, стипендии, общежитии, поступлении или контактах."
    ),
    "text_only": "Пока я обрабатываю только текстовые сообщения.",
}


def normalize(text: str) -> tuple[str, ...]:
    """Return lowercase lemmas, ignoring punctuation and whitespace."""
    doc = nlp(text.casefold())
    return tuple(
        (token.lemma_ or token.text).casefold().strip()
        for token in doc
        if not token.is_space and not token.is_punct
    )


def contains_phrase(words: Sequence[str], phrase: Sequence[str]) -> bool:
    """Check for a whole sequence of tokens, not an arbitrary substring."""
    phrase_length = len(phrase)
    if phrase_length == 0 or phrase_length > len(words):
        return False
    return any(
        tuple(words[index:index + phrase_length]) == tuple(phrase)
        for index in range(len(words) - phrase_length + 1)
    )


# Normalize the examples once so both messages and keywords use the same model.
INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (intent, normalize(keyword))
    for intent, keywords in INTENTS
    for keyword in keywords
)


def classify_intent(text: str) -> str:
    """Map a text message to an intent using Russian word lemmas."""
    cleaned = text.strip()
    command = cleaned.split(maxsplit=1)[0].split("@", maxsplit=1)[0].casefold() if cleaned else ""
    if command == "/start":
        return "start"
    if command == "/help":
        return "help"

    words = normalize(cleaned)
    for intent, pattern in INTENT_PATTERNS:
        if contains_phrase(words, pattern):
            return intent
    return "fallback"


def get_response(text: str) -> str:
    return RESPONSES[classify_intent(text)]


TEST_CASES: tuple[tuple[str, str], ...] = (
    ("Здравствуйте!", "greeting"),
    ("Доброе утро", "greeting"),
    ("/start", "start"),
    ("/help", "help"),
    ("Где посмотреть расписание занятий?", "schedule"),
    ("Покажи расписания на завтра", "schedule"),
    ("Когда экзамены?", "session"),
    ("Как записаться на пересдачу?", "session"),
    ("Какие условия получения стипендии?", "scholarship"),
    ("Можно ли заселиться в общежитие?", "dormitory"),
    ("Как поступить в университет?", "admission"),
    ("Где найти телефон деканата?", "dean_office"),
    ("Спасибо!", "thanks"),
    ("Как изменить пароль?", "fallback"),
)


def run_self_tests() -> int:
    """Run a small reproducible intent check; this is not a general accuracy score."""
    failures: list[str] = []
    for text, expected in TEST_CASES:
        actual = classify_intent(text)
        if actual != expected:
            failures.append(f"{text!r}: ожидалось {expected}, получено {actual}")

    passed = len(TEST_CASES) - len(failures)
    print(f"Проверки: {passed}/{len(TEST_CASES)} пройдены.")
    if failures:
        print("Ошибки:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Все тестовые примеры распознаны правильно.")
    return 0


def create_bot() -> telebot.TeleBot:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Задайте токен в переменной окружения TELEGRAM_BOT_TOKEN перед запуском бота."
        )

    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=["start"])
    def handle_start(message: telebot.types.Message) -> None:
        bot.reply_to(message, RESPONSES["start"])

    @bot.message_handler(commands=["help"])
    def handle_help(message: telebot.types.Message) -> None:
        bot.reply_to(message, RESPONSES["help"])

    @bot.message_handler(content_types=["text"])
    def handle_text(message: telebot.types.Message) -> None:
        if not message.text or not message.text.strip():
            bot.reply_to(message, RESPONSES["fallback"])
            return
        bot.reply_to(message, get_response(message.text))

    @bot.message_handler(
        content_types=[
            "audio", "document", "animation", "game", "photo", "sticker",
            "video", "video_note", "voice", "location", "contact", "venue",
            "dice", "web_app_data",
        ]
    )
    def handle_non_text(message: telebot.types.Message) -> None:
        bot.reply_to(message, RESPONSES["text_only"])

    return bot


def main() -> int:
    parser = argparse.ArgumentParser(description="FAQ-бот Университета «Синергия»")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="проверить распознавание примеров без подключения к Telegram",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_tests()

    bot = create_bot()
    bot.infinity_polling(skip_pending=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
