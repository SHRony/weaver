import asyncio


async def test_async_harness_works() -> None:
    await asyncio.sleep(0)
    assert True
