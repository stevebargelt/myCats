package vision

import (
	"bytes"
	"context"
	"fmt"
	"io/ioutil"
	"log"
	"time"

	"github.com/Azure/azure-sdk-for-go/services/cognitiveservices/v1.1/customvision/prediction"
	uuid "github.com/satori/go.uuid"
)

// ImagePredictor : predicts what an image contains
type ImagePredictor struct {
	Predictor   prediction.BaseClient
	ProjectID   uuid.UUID
	IterationID uuid.UUID
	// PredictionKey string
	// EndpointURL   string
	FilePath  string
	ReadDelay int
}

// Predict - takes info and returns a prediction
func (p *ImagePredictor) Predict() prediction.ImagePrediction {

	ctx := context.Background()
	var testImageData []byte
	var err error
	retryCount := 0
	//this is really UGLY. Finding that we error on the first file because it's not done writing when we read it.
	for ok := true; ok; ok = (len(testImageData) == 0) {
		testImageData, err = ioutil.ReadFile(p.FilePath)
		if err != nil {
			log.Fatal(err)
		}
		//fmt.Printf("Length %v\n", len(testImageData))
		if len(testImageData) == 0 {
			retryCount++
			time.Sleep(time.Duration(p.ReadDelay) * time.Second)
		}
	}
	log.Printf("RetryCount = %v\n", retryCount)
	results, err := p.Predictor.PredictImage(ctx, p.ProjectID, ioutil.NopCloser(bytes.NewReader(testImageData)), &p.IterationID, "")
	if err != nil {
		fmt.Println("\n\npredictor.PredictImage Failed.")
		log.Fatal(err)
	}
	return results
}
