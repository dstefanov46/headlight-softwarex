package main

import (
	"github.com/gin-gonic/gin"
	"github.com/robfig/cron/v3"
	"github.com/spf13/afero"
	plc "gitlab.eclipse.org/eclipse-research-labs/nephele-project/opencall-2/headlight/actions-handler/placements"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"log/slog"
	"net/http"
	"os"
	"sync"
)

func main() {
	batchDir, exists := os.LookupEnv("BATCH_PLACEMENTS_DIR")
	if !exists {
		slog.Error("BATCH_PLACEMENTS_DIR must be set")
		os.Exit(1)
	}

	karmadaConfig, exists := os.LookupEnv("KARMADA_CONFIG_PATH")
	if !exists {
		slog.Error("KARMADA_CONFIG_PATH must be set")
		os.Exit(1)
	}

	batchSubmitCron, exists := os.LookupEnv("BATCH_SUBMIT_CRON")
	if !exists {
		slog.Error("BATCH_SUBMIT_CRON must be set")
		os.Exit(1)
	}

	config, err := clientcmd.BuildConfigFromFlags("", karmadaConfig)
	if err != nil {
		slog.Error("Failed to build config from flags: ", "err", err)
		return
	}

	gv := schema.GroupVersion{Group: "policy.nephele-project.eu", Version: "v1alpha1"}
	config.GroupVersion = &gv
	config.APIPath = "/apis"

	var Scheme = runtime.NewScheme()
	var Codecs = serializer.NewCodecFactory(Scheme)
	config.NegotiatedSerializer = Codecs.WithoutConversion()

	restClient, err := rest.RESTClientFor(config)
	if err != nil {
		slog.Error("Failed to get REST client: ", "err", err)
		return
	}

	mutex := &sync.Mutex{}
	fs := afero.NewOsFs()
	placementsService := plc.New(mutex, restClient, fs, batchDir)

	r := gin.Default()
	r.POST("/placements", postPlacements(placementsService))

	go func() {
		slog.Info("Starting server...")
		if err := r.Run(":8080"); err != nil {
			slog.Error("Failed to start server: ", "err", err)
			os.Exit(1)
		}
	}()

	c := cron.New()
	_, err = c.AddFunc(batchSubmitCron, placementsService.SubmitBatch)
	slog.Info("Starting batch updates submitter...")
	if err != nil {
		slog.Error("Failed to start batch updates submitter: ", "err", err)
		os.Exit(1)
	}
	c.Start()

	// Block forever.
	select {}
}

func postPlacements(placementService *plc.Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var placements []plc.Placement

		if err := c.BindJSON(&placements); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON body provided."})
			return
		}

		if err := placementService.AddToBatch(placements); err != nil {
			slog.Error("Failed to batch placement updates: ", "err", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to batch placement updates."})
			return
		}

		c.JSON(http.StatusAccepted, gin.H{"message": "Placement updates batched."})
	}
}
