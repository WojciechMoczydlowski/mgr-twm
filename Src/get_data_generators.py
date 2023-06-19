from keras.preprocessing.image import ImageDataGenerator
from configparser import ConfigParser


def GetDataGenerators(
    augmentation_parameters,
    train_images_path=None,
    train_targets_path=None,
    test_images_path=None,
    test_targets_path=None,
    batch_size=64,
    seed=42,
):
    generators = []

    def get_boolean(string):
        if string == "True":
            return True
        elif string == "False":
            return False

    if train_images_path and train_targets_path:
        train_datagen = ImageDataGenerator(
            featurewise_center=get_boolean(
                augmentation_parameters["featurewise_center"]
            ),
            samplewise_center=get_boolean(augmentation_parameters["samplewise_center"]),
            featurewise_std_normalization=get_boolean(
                augmentation_parameters["featurewise_std_normalization"]
            ),
            samplewise_std_normalization=get_boolean(
                augmentation_parameters["samplewise_std_normalization"]
            ),
            zca_whitening=get_boolean(augmentation_parameters["zca_whitening"]),
            zca_epsilon=float(augmentation_parameters["zca_epsilon"]),
            rotation_range=float(augmentation_parameters["rotation_range"]),
            width_shift_range=float(augmentation_parameters["width_shift_range"]),
            height_shift_range=float(augmentation_parameters["height_shift_range"]),
            shear_range=float(augmentation_parameters["shear_range"]),
            zoom_range=float(augmentation_parameters["zoom_range"]),
            channel_shift_range=float(augmentation_parameters["channel_shift_range"]),
            fill_mode=augmentation_parameters["fill_mode"],
            cval=float(augmentation_parameters["cval"]),
            horizontal_flip=get_boolean(augmentation_parameters["horizontal_flip"]),
            vertical_flip=get_boolean(augmentation_parameters["vertical_flip"]),
            rescale=float(augmentation_parameters["rescale"]),
            validation_split=float(augmentation_parameters["validation_split"]),
        )

        train_image_generator = train_datagen.flow_from_directory(
            directory=train_images_path,
            batch_size=batch_size,
            class_mode=None,
            subset="training",
            seed=seed,
        )

        train_target_generator = train_datagen.flow_from_directory(
            directory=train_targets_path,
            batch_size=batch_size,
            class_mode=None,
            subset="training",
            seed=seed,
        )

        validation_image_generator = train_datagen.flow_from_directory(
            directory=train_images_path,
            batch_size=batch_size,
            class_mode=None,
            subset="validation",
            seed=seed,
        )

        validation_target_generator = train_datagen.flow_from_directory(
            directory=train_targets_path,
            batch_size=batch_size,
            class_mode=None,
            subset="validation",
            seed=seed,
        )

        train_generator = zip(train_image_generator, train_target_generator)
        validation_generator = zip(
            validation_image_generator, validation_target_generator
        )

        generators.extend([train_generator, validation_generator])

    if test_images_path and test_targets_path:
        test_datagen = ImageDataGeneratorImageDataGenerator(
            featurewise_center=get_boolean(
                augmentation_parameters["featurewise_center"]
            ),
            samplewise_center=get_boolean(augmentation_parameters["samplewise_center"]),
            featurewise_std_normalization=get_boolean(
                augmentation_parameters["featurewise_std_normalization"]
            ),
            samplewise_std_normalization=get_boolean(
                augmentation_parameters["samplewise_std_normalization"]
            ),
            zca_whitening=get_boolean(augmentation_parameters["zca_whitening"]),
            zca_epsilon=float(augmentation_parameters["zca_epsilon"]),
            rotation_range=float(augmentation_parameters["rotation_range"]),
            width_shift_range=float(augmentation_parameters["width_shift_range"]),
            height_shift_range=float(augmentation_parameters["height_shift_range"]),
            shear_range=float(augmentation_parameters["shear_range"]),
            zoom_range=float(augmentation_parameters["zoom_range"]),
            channel_shift_range=float(augmentation_parameters["channel_shift_range"]),
            fill_mode=augmentation_parameters["fill_mode"],
            cval=float(augmentation_parameters["cval"]),
            horizontal_flip=get_boolean(augmentation_parameters["horizontal_flip"]),
            vertical_flip=get_boolean(augmentation_parameters["vertical_flip"]),
            rescale=float(augmentation_parameters["rescale"]),
            validation_split=float(augmentation_parameters["validation_split"]),
        )

        test_image_generator = test_datagen.flow_from_directory(
            directory=test_images_path,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
            seed=seed,
        )

        test_target_generator = test_datagen.flow_from_directory(
            directory=test_targets_path,
            batch_size=batch_size,
            class_mode=None,
            shuffle=False,
            seed=seed,
        )

        test_generator = zip(test_image_generator, test_target_generator)

        generators.append(test_generator)

    if generators:
        return generators

    else:
        print("Invalid Input for Data Paths. Plese Check and Retry.")
        return None


config = ConfigParser()
test_generator = GetDataGenerators(
    test_images_path="./Data/Test/Images",
    test_targets_path="./Data/Test/Masks",
    train_images_path="./Data/Train/Masks",
    train_targets_path="./Data/Train/Masks",
    batch_size=128,
    seed=42,
)
