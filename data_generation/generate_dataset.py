# This script generates a dataset for training the hippocampal model based on the provided parameters
# Parameters:
# Dimension: The dimensions of the input tensor given as comma-separated, e.g., "1,2,3,4" is a 1 row, 2 column set of units where each unit has 3 rows and 4 columns
# Units: Which units to fill in (doesn't really seem to be a good way to specify this?) Possible values: H(orizontal), V(ertical)
# Sparsity: Relative sparsity for each unit, encoded as decimal in [0,1]
# Type: Train, Test

import sys,getopt
import csv

DIMENSION_ARG ='-d'
UNITS_ARG = '-u'
SPARSITY_ARG = '-s'
DATATYPE_ARG = '-t'
FILENAME_ARG = '-f'

EMERGENT_HEADER = '_H:'
EMERGENT_NAME_COLUMN = '$Name'
EMERGENT_DATA = '_D:'
EMERGENT_INPUT = '%Input'

# create the <#:#,#,#,#> string representing the size of the tensor for the first column 
def format_tensor_header(dimensions):

    dims = ""
    for i in range(len(dimensions) - 1):
        dims += dimensions[i]
        dims += ','
    dims += dimensions[-1]

    header = '<{}:{}>'.format(len(dimensions), dims)

    return header

def format_column(dimensions):
    """Format the input coordinate column header

    dimensions: the coordinate to format
    """
    column = "{}[{}:".format(EMERGENT_INPUT, str(len(dimensions)))

    for i in range(len(dimensions) - 1):
        column+= (str(dimensions[i]) + ',')

    column += str(dimensions[i])
    column += ']'


    return column

# append the normal column headers
def create_normal_columns(header, dimensions, previous_coordinate):
        """Create the normal column headers

        header: the list storing each column header, in order
        dimensions: the overall tensor dimensions
        previous_coordinate: the last coordinate string added to the list
        """

        if previous_coordinate == dimensions:
            return
        else:
            # add column
            print("whee")
            # increment coordinate

            # recurse 



def create_header(dimensions):

        header = []
        header.append(EMERGENT_HEADER)
        header.append(EMERGENT_NAME_COLUMN)
        
        # setup the first header column
        first_coordinate = [0 for i in dimensions]
        first_column = format_column(first_coordinate)
        first_column += str(format_tensor_header(dimensions))

        header.append(first_column) 

        # setup the columns for all further dimensions

        return header

def write_file(filename, dimensions, units, sparsity, datatype):

    # get actual numbers, not string
    tensor_dimensions = dimensions.split(',')

    with open(filename, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header = create_header(tensor_dimensions)
        print("Header: ", header)

        data_writer.writerow(header)


def main(argv):
    # do stuff

    try:
        opts, args = getopt.getopt(argv, "f:d:u:s:t:")
    except getopt.GetoptError:
        print ('Fix arguments')
        print ('generate_dataset.py -f <filename> -d <dimensions> -u <units> -s <sparsity> -t <type>')
        sys.exit(2)

    filename = ""
    dimensions = ""
    units = ""
    sparsity = ""
    dataType = ""
   
    for opt, arg in opts:
        if opt == DIMENSION_ARG:
            dimensions = arg
        elif opt == UNITS_ARG:
            units = arg
        elif opt == SPARSITY_ARG:
            sparsity = arg
        elif opt == DATATYPE_ARG:
            dataType = arg
        elif opt == FILENAME_ARG:
            filename = arg

    print("Arguments: ", filename, dimensions, units, sparsity, dataType)

    write_file(filename, dimensions, units, sparsity, dataType)


if __name__ == "__main__":
    main(sys.argv[1:])
