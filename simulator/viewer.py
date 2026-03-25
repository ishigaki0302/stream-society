"""Viewer agent for the stream-society simulator."""

from __future__ import annotations

import random
import uuid
from typing import Optional

from .schemas import CommentCandidate, Persona, ViewerState

# Comment templates by communication_style
_TEMPLATES: dict[str, list[str]] = {
    "friendly": [
        "わあ、{topic}の話、めちゃくちゃ面白い！",
        "{topic}について詳しく教えてほしいな〜",
        "今日も配信来てよかった！{topic}最高！",
        "ありがとうございます！{topic}大好きです！",
        "すごく楽しんでます！{topic}もっと聞きたい！",
    ],
    "analytical": [
        "{topic}の観点から考えると、興味深い点がいくつかありますね。",
        "{topic}についてのデータや根拠はどのようなものでしょうか？",
        "先ほどの{topic}の説明、論理的でわかりやすかったです。",
        "{topic}に関しては、複数のアプローチが考えられますよね。",
        "その{topic}の仮説を検証する方法を考えてみたのですが…",
    ],
    "enthusiastic": [
        "{topic}!!!!! 最高すぎる！！！！",
        "うおおおお{topic}きたああああ！！",
        "待ってた！！{topic}の話！超テンション上がる！！",
        "{topic}が好きすぎてヤバい！！！もっと語って！！",
        "やっっっっば！{topic}めちゃくちゃ好き！！！！",
    ],
    "quiet": [
        "{topic}、良いですね。",
        "なるほど、{topic}ですか。",
        "ありがとうございます。",
        "{topic}について少し気になりました。",
        "参考になりました。",
    ],
    "talkative": [
        "あ、{topic}といえば、私も先日体験したんですけど、それがもう本当に面白くて、友達にも勧めたんですよ！",
        "{topic}の話を聞いてたら色々思い出してきた！昔からずっと好きで、特に〇〇のところが大好きで…",
        "ねえねえ、{topic}って最近どう？私的にはすごく注目してて、毎日チェックしてるんだけど！",
        "{topic}、私も詳しいよ！色々語れるから聞いて！特に最近のトレンドとか面白いよね〜",
        "配信最高！{topic}の話しながら他のこともいっぱい話してほしい！あと質問もあって…",
    ],
    "critical": [
        "{topic}については少し懐疑的です。本当にそうなのでしょうか？",
        "その{topic}の見方は偏っていませんか？別の視点もあると思いますが。",
        "{topic}、一般的にそう言われてますが、実際はどうなんでしょうね。",
        "{topic}に関する情報の信頼性は確認されていますか？",
        "{topic}、個人的にはあまり同意できない部分もあります。",
    ],
}

_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "friendly": [
        "{topic}って初心者でも楽しめますか？",
        "{topic}のおすすめを教えてもらえますか？",
        "{topic}はどこで始めればいいですか？",
    ],
    "analytical": [
        "{topic}の具体的なメリットとデメリットを教えていただけますか？",
        "{topic}において最も重要な要素は何だと思いますか？",
        "{topic}の将来性についてどのようにお考えですか？",
    ],
    "enthusiastic": [
        "{topic}の一番熱い部分ってどこですか！？！",
        "{topic}でおすすめの最強コンテンツ教えて！！",
        "{topic}、もっと深く掘り下げてほしい！何から始めればいい！？",
    ],
    "quiet": [
        "{topic}について教えていただけますか？",
        "{topic}は難しいですか？",
        "{topic}のコツはありますか？",
    ],
    "talkative": [
        "{topic}って色々あるじゃないですか、その中でどれが一番おすすめですか？あと理由も教えてほしい！",
        "{topic}のことをもっと詳しく知りたくて！特に初心者向けのアドバイスとか、どうすれば上手くなれるかとか！",
        "{topic}についていくつか質問があるんですけど、時間ありますか？",
    ],
    "critical": [
        "{topic}のデメリットについてはどのようにお考えですか？",
        "{topic}に批判的な意見もあると思いますが、それについてはどう思いますか？",
        "{topic}の欠点を正直に教えていただけますか？",
    ],
}

_TOPIC_SENTIMENTS: dict[str, float] = {
    "gaming": 0.6,
    "music": 0.7,
    "anime": 0.65,
    "technology": 0.5,
    "lifestyle": 0.55,
    "cooking": 0.7,
    "travel": 0.75,
    "sports": 0.6,
    "fashion": 0.6,
    "unknown": 0.3,
}


class ViewerAgent:
    """Simulates a single viewer in the livestream."""

    def __init__(self, persona: Persona, viewer_id: str, seed: int) -> None:
        self.persona = persona
        self.viewer_id = viewer_id
        self.seed = seed
        self._comment_count = 0

        # Initialize interest_by_topic from persona interests
        interest_map: dict[str, float] = {}
        for interest in persona.interests:
            interest_map[interest] = 0.8
        # Default topics
        for topic in ["gaming", "music", "anime", "technology", "lifestyle"]:
            if topic not in interest_map:
                interest_map[topic] = 0.2

        self.state = ViewerState(
            viewer_id=viewer_id,
            persona_id=persona.persona_id,
            persona_group=persona.persona_group,
            interest_by_topic=interest_map,
            affinity_to_streamer=0.5,
            activity_level=persona.base_activity_level,
            emotion_state="neutral",
            participation_style="lurker" if persona.base_activity_level < 0.4 else "active",
        )

    def decide_comment(
        self,
        turn: int,
        streamer_topic: str,
        recent_response: Optional[str],
    ) -> Optional[CommentCandidate]:
        """Decide whether to comment and generate a candidate if so."""
        rng = random.Random(self.seed + turn * 1000 + hash(self.viewer_id) % 10000)

        # Base probability from activity level
        prob = self.state.activity_level

        # Boost if topic matches interests
        topic_interest = self.state.interest_by_topic.get(streamer_topic, 0.2)
        if topic_interest > 0.5:
            prob *= 1.3

        # Boost from affinity
        prob *= 0.7 + 0.6 * self.state.affinity_to_streamer

        # Clamp
        prob = min(prob, 0.95)

        if rng.random() >= prob:
            return None

        # Generate comment
        self._comment_count += 1
        is_question = rng.random() < 0.25

        text = self._generate_text(rng, streamer_topic, is_question)
        sentiment = self._compute_sentiment(rng, streamer_topic)
        toxicity = self._compute_toxicity(rng)
        novelty = self._compute_novelty(rng, turn)

        comment_id = str(uuid.uuid4())[:8]

        return CommentCandidate(
            comment_id=comment_id,
            viewer_id=self.viewer_id,
            persona_id=self.persona.persona_id,
            persona_group=self.persona.persona_group,
            text=text,
            timestamp_turn=turn,
            topic=streamer_topic,
            sentiment=sentiment,
            question_flag=is_question,
            toxicity_score=toxicity,
            novelty_score=novelty,
        )

    def _generate_text(self, rng: random.Random, topic: str, is_question: bool) -> str:
        """Generate template-based comment text."""
        style = self.persona.communication_style
        if style not in _TEMPLATES:
            style = "friendly"

        if is_question:
            templates = _QUESTION_TEMPLATES.get(style, _QUESTION_TEMPLATES["friendly"])
        else:
            templates = _TEMPLATES.get(style, _TEMPLATES["friendly"])

        template = rng.choice(templates)
        return template.format(topic=topic)

    def _compute_sentiment(self, rng: random.Random, topic: str) -> float:
        """Compute sentiment score based on topic and persona."""
        base = _TOPIC_SENTIMENTS.get(topic, 0.3)
        style = self.persona.communication_style
        if style == "enthusiastic":
            base += 0.2
        elif style == "critical":
            base -= 0.3
        elif style == "friendly":
            base += 0.1

        # Add affinity influence
        base += (self.state.affinity_to_streamer - 0.5) * 0.2

        # Add noise
        noise = rng.gauss(0, 0.1)
        return max(-1.0, min(1.0, base + noise))

    def _compute_toxicity(self, rng: random.Random) -> float:
        """Compute toxicity score (low by default)."""
        style = self.persona.communication_style
        base = 0.02
        if style == "critical":
            base = 0.08
        noise = rng.uniform(0, 0.05)
        return min(1.0, base + noise)

    def _compute_novelty(self, rng: random.Random, turn: int) -> float:
        """Compute novelty score (higher in early turns)."""
        decay = max(0.2, 1.0 - self._comment_count * 0.1)
        base = rng.uniform(0.3, 0.8) * decay
        return max(0.0, min(1.0, base))

    def update_state(self, was_selected: bool, streamer_response: Optional[str]) -> None:
        """Update viewer state after a turn."""
        if was_selected:
            # Affinity boost when selected
            self.state.affinity_to_streamer = min(1.0, self.state.affinity_to_streamer + 0.05)
            self.state.emotion_state = "happy"
            self.state.participation_style = "active"
        else:
            # Slight affinity decay
            self.state.affinity_to_streamer = max(0.0, self.state.affinity_to_streamer - 0.01)
            if self.state.affinity_to_streamer < 0.3:
                self.state.emotion_state = "bored"
            else:
                self.state.emotion_state = "neutral"

        # Activity level drift
        if self.state.affinity_to_streamer > 0.7:
            self.state.activity_level = min(1.0, self.state.activity_level + 0.02)
        else:
            self.state.activity_level = max(0.1, self.state.activity_level - 0.01)
