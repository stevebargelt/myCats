package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"time"

	uuid "github.com/satori/go.uuid"
	"github.com/spf13/viper"
	"gopkg.in/fsnotify.v1"

	firebase "firebase.google.com/go"
	"google.golang.org/api/option"

	"github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.1/customvision/prediction"
)

type LitterboxUser struct {
	Name        string
	Probability float64
}

func main() {

	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	// viper.AddConfigPath("/etc/catPredictor/")  // path to look for the config file in
	// viper.AddConfigPath("$HOME/.catPredictor") // call multiple times to add many search paths
	viper.AddConfigPath(".")    // look for config in the working directory
	err := viper.ReadInConfig() // Find and read the config file
	if err != nil {             // Handle errors reading the config file
		panic(fmt.Errorf("Fatal error config file: %s \n", err))
	}

	var (
		projectIDString      = viper.GetString("CUSTOM_VISION_PROJECT_ID")
		predictionKey        = viper.GetString("CUSTOM_VISION_PREDICTION_KEY")
		predictionResourceID = viper.GetString("CUSTOM_VISION_RESOURCE_ID")
		endpointURL          = viper.GetString("CUSTOM_VISION_ENDPOINT")
		iterationIDString    = viper.GetString("CUSTOM_VISION_ITERATION_ID")
		watchFolder          = viper.GetString("WATCH_FOLDER")
	)

	if projectIDString == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_PROJECT_ID environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**")
	}

	if predictionKey == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_PREDICTION_KEY environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**\n")
	}

	if predictionResourceID == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_RESOURCE_ID environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**\n")
	}

	if endpointURL == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_ENDPOINT environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**")
	}

	if iterationIDString == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_ITERATION_ID environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**")
	}

	projectID, err := uuid.FromString(projectIDString)
	if err != nil {
		fmt.Printf("Something went wrong creating ProjectID UUID: %s", err)
	}

	iterationID, err := uuid.FromString(iterationIDString)
	if err != nil {
		fmt.Printf("Something went wrong creating Iteration UUID: %s", err)
	}

	var litterboxPicSet []LitterboxUser

	// creates a new file watcher
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		log.Fatal(err)
	}
	defer watcher.Close()

	done := make(chan bool)
	timeout := make(chan bool, 1)

	go func() {
		for {
			select {
			case event := <-watcher.Events:
				//log.Println("event:", event)
				if event.Op&fsnotify.Create == fsnotify.Create {
					log.Println("created file:", event.Name) //event.Name is the file & path
					results := predict(predictionKey, endpointURL, projectID, iterationID, event.Name)
					highestProbabilityTag := processResults(results)
					litterboxPicSet = append(litterboxPicSet, highestProbabilityTag)
					// TODO: If this is the first photo then set a timer so we don't wait indef for 5 photos...
					if len(litterboxPicSet) == 1 {
						go func() {
							time.Sleep(15 * time.Second)
							timeout <- true
						}()
					}
					// Pic the best of the set of 5 pics
					if len(litterboxPicSet) == 5 {
						litterboxUser, weHaveCat := determineResults(litterboxPicSet)
						litterboxPicSet = nil
						if weHaveCat {
							fmt.Printf("I am %v sure that %s used the catbox!\n", litterboxUser.Probability*100, litterboxUser.Name)
						} else {
							fmt.Printf("I am %v sure that we had a false motion event!\n", litterboxUser.Probability*100)
						}
					}
				}
			case <-timeout:
				fmt.Println("Timed Out")

			case err := <-watcher.Errors:
				log.Println("error:", err)
			}
		}
	}()

	err = watcher.Add(watchFolder)
	if err != nil {
		log.Fatal(err)
	}
	<-done

}

func processResults(results prediction.ImagePrediction) LitterboxUser {

	litterboxUser := LitterboxUser{"Negative", 0.00}
	for _, prediction := range *results.Predictions {
		fmt.Printf("\t%s: %.2f%%", *prediction.TagName, *prediction.Probability*100)
		fmt.Println("")
		//of the tags in the model pick the highest probability
		if *prediction.Probability > litterboxUser.Probability {
			litterboxUser.Name = *prediction.TagName
			litterboxUser.Probability = *prediction.Probability
		}
	}
	return litterboxUser
}

func determineResults(litterboxPicSet []LitterboxUser) (LitterboxUser, bool) {
	var highestCatIndex int
	var highestCatProbability = 0.0
	var highestNegProbability = 0.0
	var highestNegIndex int
	var weHaveCat = false
	fmt.Printf("litterboxPicSet: %v\n", litterboxPicSet)
	for index, element := range litterboxPicSet {
		if element.Name != "Negative" {
			if element.Probability > highestCatProbability {
				highestCatProbability = element.Probability
				highestCatIndex = index
				weHaveCat = true
			}
		} else {
			if element.Probability > highestNegProbability {
				highestNegProbability = element.Probability
				highestNegIndex = index
			}
		}
	}
	if weHaveCat {
		return litterboxPicSet[highestCatIndex], weHaveCat
	}
	return litterboxPicSet[highestNegIndex], weHaveCat
}

func predict(predictionKey string, endpointURL string, projectID uuid.UUID, iterationID uuid.UUID, filepath string) prediction.ImagePrediction {

	ctx := context.Background()
	fmt.Println("Predicting...")
	predictor := prediction.New(predictionKey, endpointURL)

	testImageData, err := ioutil.ReadFile(filepath)
	if err != nil {
		log.Fatal(err)
	}

	results, err := predictor.PredictImage(ctx, projectID, ioutil.NopCloser(bytes.NewReader(testImageData)), &iterationID, "")
	if err != nil {
		fmt.Println("\n\npredictor.PredictImage Failed.")
		log.Fatal(err)
	}
	return results
}

// Next Steps?
func addLitterBoxTripToFirestore(user LitterboxUser) {
	ctx := context.Background()
	sa := option.WithCredentialsFile("path/to/serviceAccount.json")
	app, err := firebase.NewApp(ctx, nil, sa)
	if err != nil {
		log.Fatalln(err)
	}

	client, err := app.Firestore(ctx)
	if err != nil {
		log.Fatalln(err)
	}
	defer client.Close()

	_, _, err = client.Collection("users").Add(ctx, user)
	if err != nil {
		log.Fatalf("Failed adding litterbox trip: %v", err)
	}

}

func writeUserToFirestore(user LitterboxUser) {

	url := "https://us-central1-myupside-65eb1.cloudfunctions.net/databaseUserAddUpdate"
	fmt.Println("Calling: ", url)

	client := http.Client{}

	userJSON, _ := json.Marshal(user)

	req, _ := http.NewRequest("POST", url, bytes.NewBuffer(userJSON))
	req.Header.Set("Content-Type", "application/json")
	res, _ := client.Do(req)

	io.Copy(os.Stdout, res.Body)

}

func TestMyCats() {
	fmt.Println("This is only a test!")
}
