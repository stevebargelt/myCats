package config

// Configuration for app
type Configuration struct {
	ProjectIDString     string `mapstructure:"CUSTOM_VISION_PROJECT_ID"`
	PredictionKey       string `mapstructure:"CUSTOM_VISION_ITERATION_ID"`
	EndpointURL         string `mapstructure:"CUSTOM_VISION_ENDPOINT"`
	IterationIDString   string `mapstructure:"CUSTOM_VISION_PREDICTION_KEY"`
	WatchFolder         string `mapstructure:"WATCH_FOLDER"`
	FirebaseCredentials string `mapstructure:"GOOGLE_FIREBASE_CREDENTIAL_FILE"`
	PhotosInSet         int    `mapstructure:"NUMBER_PHOTOS_IN_SET"`
	TimeoutValue        int    `mapstructure:"TIMOUT"`
	ReadDelay           int    `mapstructure:"READ_DELAY"`
}
