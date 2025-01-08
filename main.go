package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"strconv"
	"time"

	agentassemble "gemini-agents/gemini-agent-assemble"

	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
)

//////////////////////////////////////
// high precision floating point agent

// calc tool description
var performCalculationTool = &genai.Tool{
	FunctionDeclarations: []*genai.FunctionDeclaration{{
		Name:        "performCalculation",
		Description: "Perform a floating point calculation for the supplied values and operator",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"valueOne": {
					Type:        genai.TypeString,
					Description: "The first floating point value as a string",
				},
				"valueTwo": {
					Type:        genai.TypeString,
					Description: "The second floating point value as a string",
				},
				"operator": {
					Type:        genai.TypeString,
					Description: "the operator for the calculation. can be one of +, -, *, /, %",
				},
			},
			Required: []string{"valueOne", "valueTwo", "operator"},
		},
	}},
}

// calc tool
func performCalculation(valueOne string, valueTwo string, operator string) string {
	log.Println("running performCalculation tool for " + valueOne + " " + operator + " " + valueTwo)
	one, _ := strconv.ParseFloat(valueOne, 64)
	two, _ := strconv.ParseFloat(valueTwo, 64)
	var result float64
	switch operator {
	case "+":
		result = one + two
	case "-":
		result = one - two
	case "*":
		result = one * two
	case "/":
		result = one / two
	case "%":
		result = math.Mod(one, two)
	default:
		log.Println("unsupported operator: " + operator)
	}
	return strconv.FormatFloat(result, 'f', -1, 64)
}

// agent initialization
func initFloatAgent(ctx context.Context) (*agentassemble.Agent, error) {
	system := `Your task is to perform high precision floating point calculations.
Reply ONLY with the calculated result.`
	var tools = []*genai.Tool{performCalculationTool}
	agentFloat, err := agentassemble.InitAgent(ctx, &system, tools, callFloatTool)
	if err != nil {
		log.Println("Error initializing the float agent")
		return nil, err
	}
	return agentFloat, err
}

// tool call handler
func callFloatTool(funcall genai.FunctionCall) (string, error) {

	result := ""
	// find the function to call
	if funcall.Name == performCalculationTool.FunctionDeclarations[0].Name {
		// check the params are populated
		valueOne, exists := funcall.Args["valueOne"]
		if !exists {
			log.Fatalln("Missing value one")
		}
		valueTwo, exists := funcall.Args["valueTwo"]
		if !exists {
			log.Fatalln("Missing value two")
		}
		operator, exists := funcall.Args["operator"]
		if !exists {
			log.Fatalln("Missing value operator")
		}
		// call the calc tool
		result = performCalculation(valueOne.(string), valueTwo.(string), operator.(string))
		log.Println("calculation result: " + result)
	} else {
		log.Println("unhandled function name: " + funcall.Name)
		return "", errors.New("unhandled function name: " + funcall.Name)
	}
	return result, nil
}

// client tool for the floating point agent description
var callFloatAgentTool = &genai.Tool{
	FunctionDeclarations: []*genai.FunctionDeclaration{{
		Name:        "callFloatAgent",
		Description: "Make a request to the floating point agent. The agent will perform the calculation and return the result.",
		Parameters: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"message": {
					Type:        genai.TypeString,
					Description: "The natural language request message for the floating point calculation agent",
				},
			},
			Required: []string{"message"},
		},
	}},
}

// client tool for the floating point agent
func callFloatAgent(message string) (string, error) {
	log.Println("running callFloatAgent tool for :" + message)

	// get the float agent endpoint
	floatHostname, ok := os.LookupEnv("FLOAT_AGENT_HOSTNAME")
	if !ok {
		log.Fatalln("environment variable FLOAT_AGENT_HOSTNAME not set")
	}
	floatPort, ok := os.LookupEnv("FLOAT_AGENT_PORT")
	if !ok {
		log.Fatalln("environment variable FLOAT_AGENT_PORT not set")
	}

	// build the payload
	request := agentassemble.Request{
		Input: message,
	}
	reqDat, err := json.Marshal(request)
	if err != nil {
		return "", err
	}

	// repare the request
	req, err := http.NewRequest("POST", "http://"+floatHostname+":"+floatPort+"/agent", bytes.NewBuffer(reqDat))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	// send the post
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// extract and decode the reply
	response := agentassemble.Response{}
	respDat, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	err = json.Unmarshal(respDat, &response)
	if err != nil {
		return "", err
	}

	return response.Content, nil
}

/////////////////////
// general math agent

// agent initialization
func initMathAgent(ctx context.Context) (*agentassemble.Agent, error) {
	system := `Your task is to perform math calculations.
For floating point requests use agent tools to help with your results.
Reply ONLY with the calculated result.`
	var tools = []*genai.Tool{callFloatAgentTool}
	agentMath, err := agentassemble.InitAgent(ctx, &system, tools, callMathTool)
	if err != nil {
		log.Println("error initializing the math agent")
		return nil, err
	}
	return agentMath, err
}

// tool call handler
func callMathTool(funcall genai.FunctionCall) (string, error) {

	result := ""
	// find the function to call
	if funcall.Name == callFloatAgentTool.FunctionDeclarations[0].Name {
		// check the params are populated
		message, exists := funcall.Args["message"]
		if !exists {
			log.Println("error missing message")
			return "", errors.New("error missing message")
		}
		// call the float agent
		var err error
		result, err = callFloatAgent(message.(string))
		if err != nil {
			log.Println(err)
			return "", err
		}
		log.Println("float agent result: " + result)
	} else {
		log.Println("unhandled function name: " + funcall.Name)
		return "", errors.New("unhandled function name: " + funcall.Name)
	}
	return result, nil
}

// agent list
var agentFloat *agentassemble.Agent
var agentMath *agentassemble.Agent

// ///////////
// main entry
func main() {
	// pull in the env vars
	err := godotenv.Load()
	if err != nil {
		log.Fatalln("error loading .env file")
	}

	// initialise the float agent
	ctxFloat := context.Background()
	agentFloat, err = initFloatAgent(ctxFloat)
	if err != nil {
		log.Fatalln("error initializing the Float Agent")
	}
	defer agentFloat.Client.Close()

	// run the float agent as a service with a single session
	floatHostname, ok := os.LookupEnv("FLOAT_AGENT_HOSTNAME")
	if !ok {
		log.Fatalln("environment variable FLOAT_AGENT_HOSTNAME not set")
	}
	floatPort, ok := os.LookupEnv("FLOAT_AGENT_PORT")
	if !ok {
		log.Fatalln("environment variable FLOAT_AGENT_PORT not set")
	}
	agentFloat.NewSession()
	agentFloat.RunAgent(floatHostname, floatPort)

	time.Sleep(2000)

	// initialize the math agent
	ctxMath := context.Background()
	agentMath, err = initMathAgent(ctxMath)
	if err != nil {
		log.Fatalln("error initializing the Math Agent")
	}
	defer agentMath.Client.Close()

	// start a new math session
	agentMath.NewSession()
	// run the math agent
	result, err := agentMath.CallAgent("what is pi to 10 decimal places multiplied by 2.5")
	//result, err := agentMath.callAgent("what is 1+1")
	if err != nil {
		log.Fatalln("error calling the agent")
	}

	log.Println("result: " + result)
}
