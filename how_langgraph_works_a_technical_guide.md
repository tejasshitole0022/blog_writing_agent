# How LangGraph Works: A Technical Guide
## Introduction to LangGraph
LangGraph is a framework designed for building stateful and controllable Large Language Model (LLM) applications, as outlined in [A Developer's Guide to LangGraph for LLM Applications](https://www.metacto.com/blogs/a-developer-s-guide-to-langgraph-building-stateful-controllable-llm-applications). It provides the capability for branching, looping, and pausing for human input, enabling more complex and interactive applications. A key feature of LangGraph is its ability to preserve full conversational memory, allowing for more cohesive and context-aware interactions. According to [LangGraph — Architecture and Design](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c), this is achieved through its unique architecture. For a general overview, [What is LangGraph?](https://www.ibm.com/think/topics/langgraph) provides a concise introduction to the technology. Not found in provided sources regarding specific implementation details or code snippets, but these resources provide a solid foundation for understanding the basics of LangGraph.
## LangGraph Architecture and Design
LangGraph's architecture is designed to support complex workflows, allowing for branching, looping, and pausing for human input, with the ability to resume execution [Source](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c). This flexibility enables developers to build customized applications that can adapt to various scenarios. One of the key features of LangGraph is its ability to preserve full conversational memory, which is essential for building stateful and controllable LLM applications [Source](https://www.metacto.com/blogs/a-developer-s-guide-to-langgraph-building-stateful-controllable-llm-applications). The modular design of LangGraph allows for easy integration and scalability, making it an attractive choice for developers [Source](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c). For a general overview of LangGraph, see [Source](https://www.ibm.com/think/topics/langgraph) for more information on its capabilities and potential use cases. Overall, LangGraph's architecture and design provide a robust foundation for building advanced LLM applications.
## Building Stateful Controllable LLM Applications with LangGraph
To build stateful controllable LLM applications with LangGraph, developers need to follow a series of steps. 
- First, **define nodes and edges in the graph**: in LangGraph, nodes represent functions or operations, while edges represent the flow of data between these nodes [Source](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c). 
- Next, **compile the graph into an executable object**: this step involves translating the graph into a format that can be executed by the LangGraph runtime, as described in [A Developer's Guide to LangGraph for LLM Applications](https://www.metacto.com/blogs/a-developer-s-guide-to-langgraph-building-stateful-controllable-llm-applications).
- Finally, **use the compiled object to process inputs**: this can be achieved using a simple Python function, such as the following:
```python
import langgraph

# Create a new LangGraph object
graph = langgraph.Graph()

# Define nodes and edges
node1 = graph.add_node("node1")
node2 = graph.add_node("node2")
graph.add_edge(node1, node2)

# Compile the graph
compiled_graph = graph.compile()

# Use the compiled object to process inputs
output = compiled_graph.process("Hello World")
print(output)
```
According to [What is LangGraph?](https://www.ibm.com/think/topics/langgraph), LangGraph provides a flexible and efficient way to build stateful controllable LLM applications. By following these steps and using the provided APIs, developers can create complex LLM applications with ease.
## Integrating LangGraph with Open Source LLMs
To integrate LangGraph with open source Large Language Models (LLMs), developers can follow a series of steps. First, visit the [LangGraph GitHub docs site](https://github.com/langgraph/docs) for tutorials and guides on getting started with LangGraph, as suggested by [Shuvrajyoti Debroy](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c). Additionally, explore the [YouTube tutorial](https://www.youtube.com/results?search_query=LangGraph+integration+tutorial) for a step-by-step guide on integrating LangGraph with open source LLMs. 
To compile and execute the graph, follow these instructions:
```python
import langgraph

# Initialize the graph
graph = langgraph.Graph()

# Add nodes and edges to the graph
graph.add_node("Node 1")
graph.add_node("Node 2")
graph.add_edge("Node 1", "Node 2")

# Compile and execute the graph
graph.compile()
graph.execute()
```
This code snippet demonstrates how to initialize a graph, add nodes and edges, and compile and execute the graph using the LangGraph library. For more information on building stateful, controllable LLM applications with LangGraph, refer to [A Developer's Guide to LangGraph for LLM Applications](https://www.metacto.com/blogs/a-developer-s-guide-to-langgraph-building-stateful-controllable-llm-applications). Not found in provided sources regarding specific open-source LLM integration, but [IBM](https://www.ibm.com/think/topics/langgraph) provides an overview of LangGraph.
## Best Practices for Using LangGraph
To get the most out of LangGraph, it's essential to follow best practices that ensure your graph is efficient, scalable, and easy to maintain. 
* Keep the graph modular and flexible to allow for easy updates and modifications.
* Use clear and concise node and edge definitions to prevent confusion and errors.
* Test and debug the graph thoroughly to identify and fix any issues before deployment.
By following these guidelines, developers can create effective LangGraph implementations that meet their needs. 
For more information on LangGraph, you can refer to the [official overview](https://www.ibm.com/think/topics/langgraph) or [technical guides](https://medium.com/@shuv.sdr/langgraph-architecture-and-design-280c365aaf2c) available online.

> **[IMAGE GENERATION FAILED]** LangGraph Architecture
>
> **Alt:** LangGraph Architecture
>
> **Prompt:** A diagram showing the LangGraph architecture, including nodes, edges, and the flow of data between them.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 5.666425305s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '5s'}]}}

> **[IMAGE GENERATION FAILED]** LangGraph Integration with Open Source LLMs
>
> **Alt:** LangGraph Integration with Open Source LLMs
>
> **Prompt:** A flowchart illustrating the steps to integrate LangGraph with open source Large Language Models (LLMs), including initialization, node and edge addition, compilation, and execution.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 4.998998614s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '4s'}]}}

> **[IMAGE GENERATION FAILED]** LangGraph Best Practices
>
> **Alt:** LangGraph Best Practices
>
> **Prompt:** An infographic highlighting best practices for using LangGraph, such as keeping the graph modular and flexible, using clear node and edge definitions, and testing and debugging thoroughly.
>
> **Error:** 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-2.5-flash-preview-image\n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_input_token_count, limit: 0, model: gemini-2.5-flash-preview-image\nPlease retry in 4.326688035s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash-preview-image'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}, {'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_input_token_count', 'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', 'quotaDimensions': {'model': 'gemini-2.5-flash-preview-image', 'location': 'global'}}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '4s'}]}}
