from dynamo.sdk import DynamoContext

@service(
    name="worker",
    namespace="dynamo",
)
class Router():
    # dynamo context injected by serve
    def __init__(self, dynamo_context: DynamoContext):
        self.runtime = dynamo_context.runtime

    @async_init
    async def async_init(self):
        # e.g generate a direct client to the worker from the router
        self.worker_client = self.runtime.namespace(self.namespace).component("worker").endpoint("generate").client()

    @endpoint()
    async def generate(self, request: ChatRequest):
        # call the client to worker
        for token in self.worker_client.generate(request):
            yield token

if __name__ == "__main__":
    import asyncio
    import uvloop

    # start now injects the DynamoContext since it's a named argument in the constructor
    asyncio.run(serve(Router))

"""
# typed pydantic model that is injected by start, super explicit
class DynamoContext(BaseModel):
    runtime: DistributedRuntime
    component: Component
    endpoints: List[Endpoint]
    name: str
    namespace: str
"""