import asyncpg


class Database:
    def __init__(self, db_url):
        self._db_url = db_url
        self.pool = None

    async def connect(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(dsn=self._db_url, min_size=1, max_size=10)
            print("Database connection pool created.")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            print("Database connection pool closed.")

    async def fetch_one(self, query, *params):
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *params)