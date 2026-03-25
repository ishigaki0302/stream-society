"""Mock LLM adapter with template-based responses."""

from __future__ import annotations

import random
from typing import Dict

from .base import LLMAdapter

_RESPONSES_BY_TOPIC: dict[str, list[str]] = {
    "gaming": [
        "そのゲーム、めちゃくちゃ面白いですよね！私も最近ずっとやってます。特にボス戦が熱い！",
        "ゲームの話、盛り上がってきた！初心者の方はまずチュートリアルをしっかりやると良いですよ。",
        "それ知ってる知ってる！ゲームって本当に奥が深いよね。コミュニティも最高です！",
        "ご質問ありがとう！このゲームのコツは、焦らずじっくり進めることかな。",
        "ゲーム最高！今日も一緒に楽しみましょう！新しいアップデートも楽しみだね。",
        "その質問、いいところついてますね！ゲームの戦略については色々あって…",
    ],
    "music": [
        "音楽っていいですよね！このアーティスト、私も大好きです。ライブも最高でした！",
        "音楽の話をすると止まらない！最近のお気に入りはこれかな。",
        "そのジャンル、奥深いですよね。初めて聴く方にはこれがおすすめです！",
        "ご質問ありがとう！音楽を楽しむコツは、まず好きな曲を見つけることかな。",
        "音楽でみんなと繋がれるの、最高ですね！今日もありがとうございます！",
        "その曲、すごく良いですよね。作曲の背景も面白くて…",
    ],
    "anime": [
        "アニメ、最高ですよね！そのシリーズ、私も全話見ました！感動の連続でした。",
        "アニメの話になると熱くなっちゃう！おすすめを聞いてくれてありがとう！",
        "その作品、評判通りです！特にキャラクターの成長が素晴らしい。",
        "アニメ初心者の方には、まずこの作品から見てほしいな！",
        "ご質問ありがとう！そのアニメについては語り出すと止まらないんですが…！",
        "今期アニメ、豊作ですよね！みんなと一緒に楽しみましょう！",
    ],
    "technology": [
        "テクノロジーって本当に進化が速いですよね！最近の話題も追いかけるのが大変です。",
        "その技術、面白いですね！実際に使ってみた感想を話しますね。",
        "AI技術の発展には驚かされますよね。皆さんはどう思いますか？",
        "ご質問ありがとう！技術的な話、もっと深掘りしましょうか？",
        "最新テクノロジー、ワクワクしますね！一緒にキャッチアップしていきましょう！",
        "その観点、鋭いですね！技術トレンドの話、もっとしたいです。",
    ],
    "lifestyle": [
        "ライフスタイルの話、大切ですよね！毎日の小さな習慣が大事だと思います。",
        "そのライフハック、私も試しました！本当に効果ありましたよ。",
        "健康的な生活を送るために心がけていることを紹介しますね。",
        "ご質問ありがとう！生活の質を上げるのって、実はシンプルなことが多いんですよ。",
        "今日も配信来てくれてありがとう！一緒に良い習慣を作っていきましょう！",
        "ライフスタイル改善、一歩一歩ですね。皆さんの工夫も聞かせてください！",
    ],
}

_GENERIC_RESPONSES: list[str] = [
    "ありがとうございます！そのコメント、とても嬉しいです！",
    "良い質問ですね！もう少し詳しく話しますね。",
    "みなさんのコメント、本当に励みになります！",
    "その視点、面白いですね！考えさせられました。",
    "配信に来てくれてありがとう！一緒に楽しみましょう！",
]

_QUESTION_RESPONSES: list[str] = [
    "良い質問ありがとう！答えるね。",
    "それについては詳しく説明しますね！",
    "みんなも気になってたと思う！答えます。",
    "素晴らしい質問です！こういう質問が嬉しいです。",
    "確かに！それは大事なポイントですね。",
]


class MockLLMAdapter(LLMAdapter):
    """Template-based mock LLM adapter for simulation."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._call_count = 0

    def is_available(self) -> bool:
        """Mock adapter is always available."""
        return True

    def generate_response(
        self,
        prompt: str,
        context: Dict,
        **kwargs,
    ) -> str:
        """Generate a template-based response.

        Args:
            prompt: The prompt (used for seed variation).
            context: Should contain 'topic' and optionally 'is_question'.
            **kwargs: Ignored in mock.

        Returns:
            A template response string.
        """
        self._call_count += 1
        rng = random.Random(self._seed + self._call_count)

        topic = context.get("topic", "lifestyle")
        is_question = context.get("is_question", False)

        # Start with question acknowledgment if needed
        prefix = ""
        if is_question:
            prefix = rng.choice(_QUESTION_RESPONSES) + " "

        # Get topic-specific response
        topic_responses = _RESPONSES_BY_TOPIC.get(topic, _GENERIC_RESPONSES)
        main = rng.choice(topic_responses)

        return prefix + main
