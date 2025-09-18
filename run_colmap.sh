SCENE_NAME=our_office
DATASET_PATH=./data/$SCENE_NAME

# COLMAP
echo "Running COLMAP feature extraction and matching..."
colmap feature_extractor --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images
colmap exhaustive_matcher --database_path $DATASET_PATH/database.db
mkdir $DATASET_PATH/sparse
colmap mapper --database_path $DATASET_PATH/database.db --image_path $DATASET_PATH/images --output_path $DATASET_PATH/sparse
# colmap image_undistorter \
#     --image_path $DATASET_PATH/images \
#     --input_path $DATASET_PATH/sparse/0 \
#     --output_path $DATASET_PATH/undistorted \
#     --output_type COLMAP