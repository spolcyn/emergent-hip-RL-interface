// server.go
// implements the REST API server for the hippocampus model

package main

import (
	"fmt"
	"github.com/labstack/echo"
	"net/http"
)

type HipServer struct {
	es  *echo.Echo // echo server
	sim *Sim       // the hippocampus simulation
}

/* Creates and runs the hippocampus API server */
func (hs *HipServer) Init(address string, sim *Sim) {
	// setup properties
	hs.sim = sim
	hs.es = echo.New()

	// setup server
	hs.setupRoutes()

	// start server
	go func() { hs.es.Logger.Fatal(hs.es.Start(address)) }()
}

/*** API-Related Data Types ***/

type DatasetUpdate struct {
	Filename string `json:"filename"`
}

type TrainRequest struct {
	ToState string `json:"Training"`
}

type TestRequest struct {
	pattern string `json:"Pattern"`
}

type TestReturn struct {
	closestPatternID int
	distance         int
}

/*** END API-Related Data Types ***/

/* Setup the API endpoints */
func (hs *HipServer) setupRoutes() {

	// update training data to new dataset
	hs.es.PUT("/dataset/train/update", func(c echo.Context) error {

		// read in request
		update := new(DatasetUpdate)
		if err := c.Bind(update); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// update dataset
		err := hs.sim.RestUpdateTrainingData(update)

		if err == nil {
			return c.String(http.StatusOK, fmt.Sprintf("Training dataset updated to %v", update.Filename))
		} else {
			return c.String(http.StatusBadRequest, err.Error())
		}

	})

	// set the input pattern set for error computations
	hs.es.PUT("dataset/input/update", func(c echo.Context) error {

		// read in request
		update := new(DatasetUpdate)
		if err := c.Bind(update); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// update dataset
		hs.sim.RestUpdateInputPatternData(update)

		// finish interaction
		return c.String(http.StatusOK, fmt.Sprintf("Input pattern dataset updated to %v", update.Filename))
	})

	// test an item
	hs.es.POST("/model/testpattern", func(c echo.Context) error {

		// read in request
		tr := new(TestRequest)

		if err := c.Bind(tr); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// test the item
		str, _ := hs.sim.RestTestPattern(tr)

		// finish interaction
		return c.String(http.StatusOK, str)
	})

	// start training the model
	hs.es.PUT("/model/train", func(c echo.Context) error {

		// read in request
		tr := new(TrainRequest)

		if err := c.Bind(tr); err != nil {
			return c.String(http.StatusBadRequest, err.Error())
		}

		// update dataset
		str, _ := hs.sim.RestUpdateTrainingState(tr)

		// finish interaction
		return c.String(http.StatusOK, str)
	})
}
