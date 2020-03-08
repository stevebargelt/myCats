package main

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"cloud.google.com/go/firestore"
	uuid "github.com/satori/go.uuid"
	"github.com/spf13/viper"
	"gopkg.in/fsnotify.v1"

	firebase "firebase.google.com/go"
	"google.golang.org/api/option"

	"github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.1/customvision/prediction"
)

type LitterboxUser struct {
	Name                 string
	NameProbability      float64
	Direction            string
	DirectionProbability float64
	Photo                string
}

func main() {

	viper.SetConfigName("config")
	viper.SetConfigType("yaml")
	// viper.AddConfigPath("/etc/catPredictor/")  // path to look for the config file in
	// viper.AddConfigPath("$HOME/.catPredictor") // call multiple times to add many search paths
	viper.AddConfigPath(".") // look for config in the working directory
	err := viper.ReadInConfig()
	if err != nil {
		panic(fmt.Errorf("fatal error config file: %s ", err))
	}
	viper.SetDefault("NUMBER_PHOTOS_IN_SET", 5)
	viper.SetDefault("TIMEOUT", 15)
	viper.SetDefault("READ_DELAY", 1)

	var (
		projectIDString     = viper.GetString("CUSTOM_VISION_PROJECT_ID")
		predictionKey       = viper.GetString("CUSTOM_VISION_PREDICTION_KEY")
		endpointURL         = viper.GetString("CUSTOM_VISION_ENDPOINT")
		iterationIDString   = viper.GetString("CUSTOM_VISION_ITERATION_ID")
		watchFolder         = viper.GetString("WATCH_FOLDER")
		firebaseCredentials = viper.GetString("GOOGLE_FIREBASE_CREDENTIAL_FILE")
		photosInSet         = viper.GetInt("NUMBER_PHOTOS_IN_SET")
		timeoutValue        = viper.GetInt("TIMEOUT")
		readDelay           = viper.GetInt("READ_DELAY")
	)

	if projectIDString == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_PROJECT_ID environment variable.\n" +
			"**You may need to restart your shell or IDE after it's set.**")
	}

	if predictionKey == "" {
		log.Fatal("\n\nPlease set a CUSTOM_VISION_PREDICTION_KEY environment variable.\n" +
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

	if firebaseCredentials == "" {
		log.Fatal("\n\nPlease set a GOOGLE_FIREBASE_CREDENTIAL_FILE environment variable.\n" +
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

	predictor := prediction.New(predictionKey, endpointURL)

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
					log.Println("create file received:", event.Name) //event.Name is the file & path
					results := predict(predictor, projectID, iterationID, event.Name, readDelay)
					highestProbabilityTag := processResults(results, event.Name)
					litterboxPicSet = append(litterboxPicSet, highestProbabilityTag)
					// If this is the first photo then set a timer so we don't wait indef for 5 photos...
					if len(litterboxPicSet) == 1 {
						go func() {
							time.Sleep(time.Duration(timeoutValue) * time.Second)
							timeout <- true
						}()
					}
					// Pick the best of the set of 5 pics
					if len(litterboxPicSet) == photosInSet {
						litterboxUser, weHaveCat := determineResults(litterboxPicSet)
						doStuffWithResult(litterboxUser, firebaseCredentials, weHaveCat)
						litterboxPicSet = nil
					}
				}
			case <-timeout:
				if len(litterboxPicSet) == 0 {
					fmt.Printf("We Good. Timeout called but we processed %v pics.\n", photosInSet)
				} else if len(litterboxPicSet) > 0 {
					litterboxUser, weHaveCat := determineResults(litterboxPicSet)
					doStuffWithResult(litterboxUser, firebaseCredentials, weHaveCat)
					litterboxPicSet = nil
				} else {
					fmt.Println("Timed Out")
				}
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

func doStuffWithResult(litterboxUser LitterboxUser, firebaseCredentials string, weHaveCat bool) {

	if weHaveCat {
		fmt.Printf("I am %v%% sure that it was %s and I am ", litterboxUser.NameProbability*100, litterboxUser.Name)
		fmt.Printf("%v%% sure that they were headed %s the catbox!\n", litterboxUser.DirectionProbability*100, litterboxUser.Direction)
		addLitterBoxTripToFirestore(litterboxUser, firebaseCredentials)
	} else {
		fmt.Printf("I am %v%% sure that we had a false motion event!\n", litterboxUser.NameProbability*100)
	}
}

func processResults(results prediction.ImagePrediction, fileName string) LitterboxUser {

	litterboxUser := LitterboxUser{"Negative", 0.00, "None", 0.00, fileName}
	for _, prediction := range *results.Predictions {
		fmt.Printf("\t%s: %.2f%%\n", *prediction.TagName, *prediction.Probability*100)

		//of the tags in the model pick the highest probability
		// TODO: Use a slice for the direction, no magic strings and decouple the directions
		// TODO: well we only care about the direction if this is the highest cat, right?
		if *prediction.TagName == "in" || *prediction.TagName == "out" {
			if *prediction.Probability > litterboxUser.DirectionProbability {
				litterboxUser.Direction = *prediction.TagName
				litterboxUser.DirectionProbability = *prediction.Probability
			}
		} else if *prediction.Probability > litterboxUser.NameProbability {
			litterboxUser.Name = *prediction.TagName
			litterboxUser.NameProbability = *prediction.Probability
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
	//log.Printf("litterboxPicSet: %v\n", litterboxPicSet)
	for index, element := range litterboxPicSet {
		if element.Name != "Negative" {
			if element.NameProbability > highestCatProbability {
				highestCatProbability = element.NameProbability
				highestCatIndex = index
				weHaveCat = true
			}
		} else {
			if element.NameProbability > highestNegProbability {
				highestNegProbability = element.NameProbability
				highestNegIndex = index
			}
		}
	}
	if weHaveCat {
		return litterboxPicSet[highestCatIndex], weHaveCat
	}
	return litterboxPicSet[highestNegIndex], weHaveCat
}

func predict(predictor prediction.BaseClient, projectID uuid.UUID, iterationID uuid.UUID, filepath string, readDelay int) prediction.ImagePrediction {

	ctx := context.Background()
	fmt.Println("Predicting...")

	var testImageData []byte
	var err error
	retryCount := 0
	//this is really UGLY. Finding that we error on the first file because it's not done writing when we read it.
	for ok := true; ok; ok = (len(testImageData) == 0) {
		testImageData, err = ioutil.ReadFile(filepath)
		if err != nil {
			log.Fatal(err)
		}
		//fmt.Printf("Length %v\n", len(testImageData))
		if len(testImageData) == 0 {
			retryCount++
			time.Sleep(time.Duration(readDelay) * time.Second)
		}
	}
	log.Printf("RetryCount = %v\n", retryCount)
	results, err := predictor.PredictImage(ctx, projectID, ioutil.NopCloser(bytes.NewReader(testImageData)), &iterationID, "")
	if err != nil {
		fmt.Println("\n\npredictor.PredictImage Failed.")
		log.Fatal(err)
	}
	return results
}

// Next Steps?
func addLitterBoxTripToFirestore(user LitterboxUser, firebaseCredentials string) {
	ctx := context.Background()
	sa := option.WithCredentialsFile(firebaseCredentials)
	app, err := firebase.NewApp(ctx, nil, sa)
	if err != nil {
		log.Fatalln(err)
	}

	client, err := app.Firestore(ctx)
	if err != nil {
		log.Fatalln(err)
	}
	defer client.Close()
	//TODO: Find cat first. Add if not found?
	_, _, err = client.Collection("cats").Doc(user.Name).Collection("LitterTrips").Add(ctx, map[string]interface{}{
		"Probability":          user.NameProbability,
		"Direction":            user.Direction,
		"DirectionProbability": user.DirectionProbability,
		"Photo":                user.Photo, // right now this is the local name. Could be the URL to the photo in Cloud Storage.
		"timestamp":            firestore.ServerTimestamp,
	})
	if err != nil {
		log.Fatalf("Failed adding litterbox trip: %v", err)
	}
}

func TestMyCats() {
	fmt.Println("This is only a test!")
}
