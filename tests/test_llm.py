import asyncio
import os

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TEST") != "1",
    reason="Set RUN_LIVE_LLM_TEST=1 to run the manual, billable API check.",
)
def test_live_router():
    from code_review_agent.llm import chat, usage_log

    response = asyncio.run(
        chat([{"role": "user", "content": "Reply with the single word OK."}])
    )

    assert response.choices
    assert usage_log["calls"] >= 1
