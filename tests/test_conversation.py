from bandit.conversation import BanditConversation


def test_conversation_accumulates_each_message_once_without_private_state():
    chat = BanditConversation.start()
    chat.record_action("A")
    chat.record_feedback(-2)
    chat.record_action("B")
    chat.record_feedback(3)

    messages = chat.messages
    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant", "user"
    ]
    assert messages[1]["content"] == "A"
    assert messages[3]["content"] == "B"
    assert sum("You received" in m["content"] for m in messages) == 2
    rendered = str(messages)
    assert "p_A_true" not in rendered
    assert "cumulative_score" not in rendered

