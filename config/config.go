package config

// Configuration for app
type Configuration struct {
	ProjectID            string `mapstructure:"CUSTOM_VISION_PROJECT_ID"`
	ProjectIDDirection   string `mapstructure:"CUSTOM_VISION_PROJECT_DIRECTION_ID"`
	PredictionKey        string `mapstructure:"CUSTOM_VISION_PREDICTION_KEY"`
	EndpointURL          string `mapstructure:"CUSTOM_VISION_ENDPOINT"`
	IterationID          string `mapstructure:"CUSTOM_VISION_ITERATION_ID"`
	IterationIDDirection string `mapstructure:"CUSTOM_VISION_ITERATION_DIRECTION_ID"`
	WatchFolder          string `mapstructure:"WATCH_FOLDER"`
	FirebaseCredentials  string `mapstructure:"GOOGLE_FIREBASE_CREDENTIAL_FILE"`
	FirestoreCollection  string `mapstructure:"GOOGLE_FIRESTORE_COLLECTION"`
	PhotosInSet          int    `mapstructure:"NUMBER_PHOTOS_IN_SET"`
	TimeoutValue         int    `mapstructure:"TIMOUT"`
	ReadDelay            int    `mapstructure:"READ_DELAY"`
}
