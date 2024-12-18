package main

import (
	"context"
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
	log.Println("float: running performCalculation for " + valueOne + " " + operator + " " + valueTwo)
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
		log.Println("float: unsupported operator: " + operator)
	}
	return strconv.FormatFloat(result, 'f', -1, 64)
}

func main() {
	// pull in the env vars and api key
	err := godotenv.Load()
	if err != nil {
		log.Fatalln("Error loading .env file")
	}
	apiKey, ok := os.LookupEnv("GEMINI_API_KEY")
	if !ok {
		log.Fatalln("Environment variable GEMINI_API_KEY not set")
	}

	// initialise the float agent
	ctxFloat := context.Background()
	system := `Your task is to perform high precision floating point calculations.
Reply ONLY with the calculated result.`
	var tools = []*genai.Tool{performCalculationTool}
	clientFloat, modelFloat, err := InitAgent(ctxFloat, apiKey, &system, tools)
	if err != nil {
		log.Fatalln("Error initializing the Orchestrator")
	}
	defer clientFloat.Close()

	// start a new session
	sessionFloat := modelFloat.StartChat()

	// make the request
	resp, err := sessionFloat.SendMessage(ctxFloat, genai.Text("what is pi to 10 decimal places multiplied by 2.5"))
	if err != nil {
		log.Fatalf("Error sending message: %v", err)
	}

	// Check the response type for the first entry only for now
	part := resp.Candidates[0].Content.Parts[0]
	funcall, ok := part.(genai.FunctionCall)
	if !ok {
		// check for an ai reply
		content, ok := part.(genai.Text)
		if !ok {
			log.Fatalf("Unexpected reply type, got %T", part)
		}
		// drop out with the reply
		log.Println("Agent reply: " + content)
	}

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
		log.Println(valueOne.(string), valueTwo.(string), operator.(string))
		// call the calc tool
		result := performCalculation(valueOne.(string), valueTwo.(string), operator.(string))
		log.Println("calculation result: " + result)

		resp, err = sessionFloat.SendMessage(ctxFloat, genai.FunctionResponse{
			Name: performCalculationTool.FunctionDeclarations[0].Name,
			Response: map[string]any{
				"answer": result,
			},
		})
		if err != nil {
			log.Fatal(err)
		}
	}

	for _, part := range resp.Candidates[0].Content.Parts {
		log.Printf("Answer: %v\n", part)
	}
}

/////////
// Generic agent routines
/////////

func InitAgent(ctx context.Context, apiKey string, system *string, tools []*genai.Tool) (*genai.Client, *genai.GenerativeModel, error) {

	// create a new genai client
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, nil, err
	}
	//defer client.Close()

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
