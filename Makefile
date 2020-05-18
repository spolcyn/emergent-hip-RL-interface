# for ease of use

all: iwhipmodel

iwhipmodel: pkg/hipmodel/api.go pkg/hipmodel/util.go pkg/hipmodel/server.go pkg/hipmodel/hip.go pkg/hipmodel/params.go pkg/hipmodel/hip_utils.go cmd/iwhipmodel/main.go
	go build ./cmd/iwhipmodel

# runs the model with no GUI
# use ./iwhipmodel to run it with a GUI
# technically, using any command-line argument works
run: iwhipmodel
	./iwhipmodel nogui
