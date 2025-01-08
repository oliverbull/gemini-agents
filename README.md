# gemini-agents
## Example agents using Google Gemini in Go

This is a simple example of how to create distributed, callable LLM powered agents using Gemini and the Google Gemini SDK. The pattern is not limited to Gemini SDK, but can be applied to **any** LLM SDK or framework.

The primary assistant agent is a 'math agent' that you can ask to make basic calculations. When you ask with floating point numbers the LLM will call out to the 'float agent' on a remote end point via calling an LLM selected tool. The tool makes the API request to the remote agent waiting for the response.

The 'float agent' serves the request and passes the it to its LLM which in turn will call the perform calculation tool to process the two values via the operator. The response is then passed back to the 'math agent'. Note that the LLM transposes the request from the natural language input to specific parameters for the tool.

The 'math agent' then passses the result back to the LLM for final result.

The abstraction package (see below) allows rapid development of agents with tools through an agent specific init function, session function, and either run agent (for a service) or call agent (for direct interaction).

This is a very simple example, that can scale to many agents all interacting with each other.

## Gemini-Agent-Assemble abstraction package

The Gemini-Agent-Assemble routines display an example of how to abstract the specific SDK calls into agent specific methods and create a generalized pattern for agent creation.

**initAgent()** Creates the client, populates the model with fixed defaults, adds a system prompt if supplied, adds the tools if supplied, saves the agent 'class' parameters, and returns the agent instance

**newSession()** Starts a new session and adds to the agent 'class' parameters

**callAgent()** Runs a fixed flow (graph) of input -> loop { tool -> tool reply } -> result. This enables the LLM to call multiple tools as needed based on the input until it has all the information needed to conclude a final answer

**runAgent() & handleAgentRequest()** Starts the API service for an agent to handle external requests. All inputs are to `http://hostname:port/agent` through a POST with a basic JSON input structure. The handler calls the agent and forms the reply into a basic JSON content structure to be sent back
