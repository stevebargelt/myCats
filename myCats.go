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

	config "github.com/stevebargelt/myCats/config"
)

// LitterboxUser = defines the attributes of a cat using the litterbox
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

	var configuration config.Configuration
	err = viper.Unmarshal(&configuration)
	if err != nil {
		fmt.Printf("Unable to decode into struct, %v", err)
	}

	projectID, err := uuid.FromString(configuration.ProjectID)
	if err != nil {
		fmt.Printf("Something went wrong creating ProjectID UUID: %s", err)
	}

	projectIDDirection, err := uuid.FromString(configuration.ProjectIDDirection)
	if err != nil {
		fmt.Printf("Something went wrong creating ProjectID UUID: %s", err)
	}

	iterationID, err := uuid.FromString(configuration.IterationID)
	if err != nil {
		fmt.Printf("Something went wrong creating Iteration UUID: %s", err)
	}

	iterationIDDirection, err := uuid.FromString(configuration.IterationIDDirection)
	if err != nil {
		fmt.Printf("Something went wrong creating Iteration Direction UUID: %s", err)
	}

	predictor := prediction.New(configuration.PredictionKey, configuration.EndpointURL)

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
				if event.Op&fsnotify.Create == fsnotify.Create {
					log.Println("create file received:", event.Name) //event.Name is the file & path
					results := predict(predictor, projectID, iterationID, event.Name, configuration.ReadDelay)
					highestProbabilityTag := processCatResults(results, event.Name)
					litterboxPicSet = append(litterboxPicSet, highestProbabilityTag)
					// If this is the first photo then set a timer so we don't wait indef for 5 photos...
					if len(litterboxPicSet) == 1 {
						go func() {
							time.Sleep(time.Duration(configuration.TimeoutValue) * time.Second)
							timeout <- true
						}()
					}
					// Pick the best of the set of 5 pics
					if len(litterboxPicSet) == configuration.PhotosInSet {
						litterboxUser, weHaveCat := determineResults(litterboxPicSet)
						if weHaveCat {
							directionResults := predict(predictor, projectIDDirection, iterationIDDirection, litterboxUser.Photo, configuration.ReadDelay)
							setDirection(directionResults, &litterboxUser)
						}
						doStuffWithResult(litterboxUser, configuration.FirebaseCredentials, configuration.FirestoreCollection, weHaveCat)
						litterboxPicSet = nil
					}
				}
			case <-timeout:
				if len(litterboxPicSet) == 0 {
					fmt.Printf("We Good. Timeout called but we processed %v pics.\n", configuration.PhotosInSet)
				} else if len(litterboxPicSet) > 0 {
					litterboxUser, weHaveCat := determineResults(litterboxPicSet)
					doStuffWithResult(litterboxUser, configuration.FirebaseCredentials, configuration.FirestoreCollection, weHaveCat)
					litterboxPicSet = nil
				} else {
					fmt.Println("Timed Out")
				}
			case err := <-watcher.Errors:
				log.Println("error:", err)
			}
		}
	}()

	err = watcher.Add(configuration.WatchFolder)
	if err != nil {
		log.Fatal(err)
	}
	<-done

}

func doStuffWithResult(litterboxUser LitterboxUser, firebaseCredentials string, firestoreCollection string, weHaveCat bool) {

	if weHaveCat {
		fmt.Printf("I am %v%% sure that it was %s and I am ", litterboxUser.NameProbability*100, litterboxUser.Name)
		fmt.Printf("%v%% sure that they were headed %s the catbox!\n", litterboxUser.DirectionProbability*100, litterboxUser.Direction)
		addLitterBoxTripToFirestore(litterboxUser, firebaseCredentials, firestoreCollection)
	} else {
		fmt.Printf("I am %v%% sure that we had a false motion event!\n", litterboxUser.NameProbability*100)
	}
}

func processCatResults(results prediction.ImagePrediction, fileName string) LitterboxUser {

	//Process the results of ONE image... loop through the tag predictions in the image

	litterboxUser := LitterboxUser{"Negative", 0.00, "", 0.00, fileName}
	for _, prediction := range *results.Predictions {
		fmt.Printf("\t%s: %.2f%%\n", *prediction.TagName, *prediction.Probability*100)

		//of the tags in the model pick the highest probability
		// TODO: Use a slice or enum for the direction, no magic strings and decouple the directions
		if *prediction.Probability > litterboxUser.NameProbability {
			litterboxUser.Name = *prediction.TagName
			litterboxUser.NameProbability = *prediction.Probability
			litterboxUser.Photo = fileName
		}
	}
	return litterboxUser
}

func setDirection(directionResults prediction.ImagePrediction, litterboxUser *LitterboxUser) {

	for _, prediction := range *directionResults.Predictions {
		fmt.Printf("\t%s: %.2f%%\n", *prediction.TagName, *prediction.Probability*100)

		if *prediction.TagName == "in" || *prediction.TagName == "out" {
			if *prediction.Probability > litterboxUser.DirectionProbability {
				litterboxUser.Direction = *prediction.TagName
				litterboxUser.DirectionProbability = *prediction.Probability
			}
		}
	}
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
func addLitterBoxTripToFirestore(user LitterboxUser, firebaseCredentials string, firestoreCollection string) {
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
	_, _, err = client.Collection(firestoreCollection).Doc(user.Name).Collection("LitterTrips").Add(ctx, map[string]interface{}{
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

// TestMyCats = Test Entry Point
func TestMyCats() {
	fmt.Println("This is only a test!")
}
