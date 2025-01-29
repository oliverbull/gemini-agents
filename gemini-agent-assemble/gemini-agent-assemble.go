package geminiagentassemble

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"os"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

/////////
// Agent Assemble routines
/////////

// agent context handle
type Agent struct {
	ctx      context.Context
	Client   *genai.Client
	model    *genai.GenerativeModel
	session  *genai.ChatSession
	system   *string
	tools    []*genai.Tool
	toolCall func(funcall genai.FunctionCall) (string, error)
}

// initializer
func InitAgent(ctx context.Context, system *string, tools []*genai.Tool, toolCall func(funcall genai.FunctionCall) (string, error)) (*Agent, error) {

	// get the api key
	apiKey, ok := os.LookupEnv("GEMINI_API_KEY")
	if !ok {
		return nil, errors.New("environment variable GEMINI_API_KEY not set")
	}

	// create a new genai client
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}

	// select the model and configure to be a NL text agent
	model := client.GenerativeModel("gemini-2.0-flash-exp")
	model.SetTemperature(0)
	model.SetTopK(40)
	model.SetTopP(0.95)
	model.SetMaxOutputTokens(8192)
	if system != nil {
		model.SystemInstruction = genai.NewUserContent(genai.Text(*system))
	}
	if tools != nil {
		model.Tools = tools
	}
	model.ResponseMIMEType = "text/plain"

	// populate the agent and return
	agent := Agent{
		ctx:      ctx,
		Client:   client,
		model:    model,
		system:   system,
		tools:    tools,
		toolCall: toolCall,
	}

	return &agent, nil
}

func (agent *Agent) NewSession() {
	agent.session = agent.model.StartChat()
}

// call agent and run tools as required before returning the result
// pre-determined graph flow of request, call tools as required, return final answer
func (agent *Agent) CallAgent(message string) (string, error) {

	// check we have a session
	if agent.session == nil {
		err := errors.New("CallAgent(): no session configued. run NewSession() first")
		log.Println(err)
		return "", err
	}

	// make the initial request
	resp, err := agent.session.SendMessage(agent.ctx, genai.Text(message))
	if err != nil {
		log.Println(err)
		return "", err
	}

	// set max runs to 25
	for idx := 0; idx < 25; idx++ {
		// process each of the parts
		var funcResults []genai.Part
		for _, part := range resp.Candidates[0].Content.Parts {
			// check for a function call
			funcall, ok := part.(genai.FunctionCall)
			if ok {
				// call the agent specific handler to get the response
				result, err := agent.toolCall(funcall)
				if err != nil {
					log.Println(err)
					return "", err
				}
				// save the result in the result slice
				funcResult := genai.FunctionResponse{
					Name: funcall.Name,
					Response: map[string]any{
						"result": result,
					},
				}
				funcResults = append(funcResults, funcResult) // implicit interface cast
			}

			// check for ONLY a text answer and end here (text can be in function list)
			content, ok := part.(genai.Text)
			if len(funcResults) == 0 && ok {
				// drop out with the reply
				log.Println("agent reply: " + content)
				return string(content), nil
			}
		}

		// pass the result back to the session
		resp, err = agent.session.SendMessage(agent.ctx, funcResults...)
		if err != nil {
			log.Println(err)
			return "", err
		}
	}

	// if we are here we ran out of cycles
	return "", errors.New("message cycles exceeded")
}

// base agent request / response
type Request struct {
	Input string `json:"input"`
}
type Response struct {
	Content string `json:"content"`
}

// generalized agent request handler
func (agent *Agent) HandleAgentRequest(res http.ResponseWriter, req *http.Request) {

	// check for post
	if req.Method != "POST" {
		http.Error(res, "Bad Request", http.StatusBadRequest)
		return
	}
	// check for json mime type
	contentType := req.Header.Get("Content-Type")
	if contentType == "" || contentType != "application/json" {
		http.Error(res, "Bad Request", http.StatusBadRequest)
		return
	}
	// decode the body
	var reqBody Request
	err := json.NewDecoder(req.Body).Decode(&reqBody)
	if err != nil {
		http.Error(res, "Bad Request", http.StatusBadRequest)
		return
	}

	// call the agent
	result, err := agent.CallAgent(reqBody.Input)
	if err != nil {
		http.Error(res, "Bad Request", http.StatusBadRequest)
		return
	}

	// send the result back
	response := Response{
		Content: result,
	}
	res.Header().Set("Content-Type", "application/json")
	json.NewEncoder(res).Encode(response)
}

// generalized agent service at <hostname>:<port>/agent
func (agent *Agent) RunAgent(hostname string, port string) {
	mux := http.NewServeMux()
	mux.HandleFunc("/agent", agent.HandleAgentRequest)
	go http.ListenAndServe(hostname+":"+port, mux)
	log.Println("agent running at: " + hostname + ":" + port)
}
