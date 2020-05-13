// server.go
// implements the REST API server for the hippocampus model

package hipmodel

import (
	"encoding/json"
	"fmt"
	"github.com/labstack/echo"
	"log"
	"net/http"
	"sync"
)

type HipServer struct {
	es  *echo.Echo // echo server
	sim *Sim       // the hippocampus simulation
	mu  *sync.Mutex
}

/* Creates and runs the hippocampus API server */
func (hs *HipServer) Init(address string, sim *Sim) {
	// setup properties
	hs.sim = sim
	hs.es = echo.New()
	hs.mu = &sync.Mutex{}

	// setup server
	hs.setupRoutes()

	// setup logging
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	// start server
	go func() { hs.es.Logger.Fatal(hs.es.Start(address)) }()
}

/*** API-Related Data Types ***/

type DatasetUpdate struct {
	Source   string   `json:"method"` // source of the new dataset: "file", if reading from a file, or "body", if transmitted in request body
	Filename string   `json:"filename"`
	Patterns []string `json:"patterns"` // the patterns, with subarrays marked using ( ) to ensure proper JSON parsing
	Shape    string   `json:"shape"`    // patterns' shape. all must be the same size
}

type TestRequest struct {
	Shape            string `json:"shape"`            // the pattern's shape
	CorruptedPattern string `json:"corruptedPattern"` // the corrupted pattern's numpy json dump representation
	TargetPattern    string `json:"targetPattern"`    // the target pattern's numpy json dump representation
}

type TrainRequest struct {
	MaxRuns int `json:"maxruns" query:"maxruns" form:"maxruns"` // max runs to train for
	MaxEpcs int `json:"maxepcs" query:"maxepcs" form:"maxepcs`  // max epochs to train for
}

/*** END API-Related Data Types ***/

/* Setup the API endpoints */
func (hs *HipServer) setupRoutes() {

	// update training data to new dataset
	hs.es.PUT("/dataset/train/update", func(c echo.Context) error {

		var err error

		// read in request
		update := new(DatasetUpdate)
		if err = c.Bind(update); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		hs.mu.Lock()
		err = hs.sim.RestUpdateTrainingData(update)
		hs.mu.Unlock()

		if err == nil {
			return c.String(http.StatusOK, fmt.Sprintf("Training dataset updated from source: %v", update.Source))
		} else {
			return c.String(http.StatusBadRequest, err.Error())
		}

	})

	// test a pattern
	hs.es.POST("/model/testpattern", func(c echo.Context) error {

		// read in request
		tr := new(TestRequest)

		if err := c.Bind(tr); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// test the item
		hs.mu.Lock()
		nameError, err := hs.sim.RestTestPattern(tr)
		hs.mu.Unlock()

		// finish interaction
		if err == nil {
			jsonNE, err2 := json.Marshal(nameError)

			if err2 != nil {
				DPrintf("JSON encoding error, %v", err2.Error())
			}

			return c.String(http.StatusOK, string(jsonNE))
		} else {
			return c.String(http.StatusOK, err.Error())
		}
	})

	// start training the model
	hs.es.POST("/model/train", func(c echo.Context) error {

		// read in request
		tr := new(TrainRequest)

		if err := c.Bind(tr); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// start training
		hs.mu.Lock()
		str, err := hs.sim.RestStartTraining(tr)
		hs.mu.Unlock()

		if err == nil {
			// finish interaction
			return c.String(http.StatusOK, str)
		} else {
			return c.String(http.StatusPreconditionFailed, err.Error())
		}
	})
}
