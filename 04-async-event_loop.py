import asyncio

async def worker():
    print("Старт")

    await asyncio.sleep(2)

    print("Конец")


asyncio.run(worker())