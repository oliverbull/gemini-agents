package main

import (
	"context"
	"errors"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/google/generative-ai-go/genai"
	"github.com/joho/godotenv"
	"google.golang.org/api/option"
)

//////////////////////////////////////
// high precision floating point agent

// calc tool description
var performCalculationTool = &genai.Tool{
	FunctionDeclarations: []*genai.FunctionDeclaration{{
		Name:        "performCalculation",
		Description: "Perform a flating point calculation for the supplied values and operator",
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
	log.Println("running performCalculation for " + valueOne + " " + operator + " " + valueTwo)
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
func initFloatAgent(ctx context.Context) (*genai.Client, *genai.GenerativeModel, error) {
	system := `Your task is to perform high precision floating point calculations.
Reply ONLY with the calculated result.`
	var tools = []*genai.Tool{performCalculationTool}
	clientFloat, modelFloat, err := initAgent(ctx, &system, tools)
	if err != nil {
		log.Println("Error initializing the float agent")
		return nil, nil, err
	}
	return clientFloat, modelFloat, err
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

// client tool for the floating point agent agent

/////////////////////
// general math agent

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
	clientFloat, modelFloat, err := initFloatAgent(ctxFloat)
	if err != nil {
		log.Fatalln("error initializing the Orchestrator")
	}
	defer clientFloat.Close()

	// start a new session
	sessionFloat := modelFloat.StartChat()

	// run the agent
	result, err := callAgent(ctxFloat, sessionFloat, "what is pi to 10 decimal places multiplied by 2.5", callFloatTool)
	if err != nil {
		log.Fatalln("error calling the agent")
	}

	log.Println("result: " + result)
}

/////////
// Generic agent routines
/////////

// initializer
func initAgent(ctx context.Context, system *string, tools []*genai.Tool) (*genai.Client, *genai.GenerativeModel, error) {

	// get the api key
	apiKey, ok := os.LookupEnv("GEMINI_API_KEY")
	if !ok {
		return nil, nil, errors.New("environment variable GEMINI_API_KEY not set")
	}

	// create a new genai client
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, nil, err
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

	return client, model, nil
}

// call agent and run tools as required before returning the result
// pre-determined graph flow of request, call tools as required, return final answer
func callAgent(ctx context.Context, session *genai.ChatSession, message string, toolCall func(funcall genai.FunctionCall) (string, error)) (string, error) {

	// make the initial request
	resp, err := session.SendMessage(ctx, genai.Text(message))
	if err != nil {
		log.Println(err)
		return "", err
	}

	// set max runs to 25
	for idx := 0; idx < 25; idx++ {
		// extract the first entry only for now
		part := resp.Candidates[0].Content.Parts[0]

		// check for a function call
		funcall, ok := part.(genai.FunctionCall)
		if ok {
			// call the agent specific handler to get the response
			result, err := toolCall(funcall)
			if err != nil {
				log.Println(err)
				return "", err
			}

			// pass the result back to the session
			resp, err = session.SendMessage(ctx, genai.FunctionResponse{
				Name: funcall.Name,
				Response: map[string]any{
					"result": result,
				},
			})
			if err != nil {
				log.Println(err)
				return "", err
			}
		}

		// check for an text answer and end here
		content, ok := part.(genai.Text)
		if ok {
			// drop out with the reply
			log.Println("agent reply: " + content)
			return string(content), nil
		}
	}

	// if we are here we ran out of cycles
	return "", errors.New("message cycles exceeded")
}
