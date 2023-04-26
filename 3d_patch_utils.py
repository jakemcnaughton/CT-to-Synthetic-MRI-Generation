import os
import re
import time
import copy
import pickle
import socket
import random
import imageio
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from keras.utils import Sequence
from keras.callbacks import TensorBoard
#from sklearn.preprocessing import StandardScaler
from scipy.ndimage.interpolation import zoom
from skimage import exposure



class CategoriseNiftis():
    """A class for categorising niftis segmented by SPM12 in MatLab. This
    will create class atributes raw, seg_1, seg_2 and seg_3 for original
    scan, white matter, grey matter, and CSF segmentations respectively. """

    def __init__(self, path, require_oasis=False, require_string=".", exclude_string=None):
        self.path = path

        files = os.listdir(self.path)
        files.sort()

        self.raw = []
        self.seg_1 = []
        self.seg_2 = []
        self.seg_3 = []
        self.seg_4 = []

        # TODO: rewrite more concisely as list comprehensions
        for file in files:

            if (file.endswith(".mat") or
                file.endswith(".json") or
                file.endswith(".jsn")):
                continue

            if "OAS3" not in file and require_oasis:
                continue

            if require_string not in file:
                continue

            if exclude_string and exclude_string in file:
                continue

            # TODO: rewrite this to be more modular / useful for any user
            # Broken or misregistered images, e.g. necks not heads, from
            #  author's specific dataset.
            if "OAS30288" in file or "OAS30038" in file \
                or "OAS30131" in file or "OAS30581" in file\
                or "OAS30920" in file or "OAS30001" in file:
                continue

            suffix = file[:2]
            full_path = os.path.join(self.path, file)
            if suffix == "c1":
                self.seg_1.append(full_path)
            elif suffix == "c2":
                self.seg_2.append(full_path)
            elif suffix == "c3":
                self.seg_3.append(full_path)
            elif suffix == "c4":
                self.seg_4.append(full_path)
            elif suffix == "c5":
                pass
            elif suffix == "wc":
                pass
            else:
                self.raw.append(full_path)

    def three_segs(self, first_volume = 0):

        three_segs_object = [self.seg_1[first_volume:],
                      self.seg_2[first_volume:],
                      self.seg_3[first_volume:]]

        return three_segs_object

class Patcher():
    """A convenience class used by patch wise data generators.
    
    Will take an equivalent patch from any supplied volume.

    Example:

    patcher = Patcher([i, j, k], self.patch_size)
    t1_patch = patcher.patch(t1_volume)
    seg_patch = patcher.patch(seg)
    
    """

    def __init__(self, indices, patch_size):
        self.idcs = indices
        self.patch_size = patch_size

    def patch(self, volume):
        patch = volume[self.idcs[0]: self.idcs[0] + self.patch_size,
                       self.idcs[1]: self.idcs[1] + self.patch_size,
                       self.idcs[2]: self.idcs[2] + self.patch_size, ]

        return patch


class PatchSequence(Sequence):
    """Keras generator that takes a list of paths to niftis and
    corresponding segmentations, either in SPM format (ie c1_..,
    c2_.., c3_..) or BraTS format. """

    def __init__(self, x_lists, y_lists, batch_size,
                 patch_size, stride, volumes_to_analyse, first_vol=0, unet=True,
                 validation=False, secondary_input=True, spatial_offset=0,
                 randomise_spatial_offset=False, reslice_isotropic=0, x_only=False,
                 prediction=False, shuffle=True, verbose=True, BraTS_like_y=False,
                 histogram_equalise=False, CLAHE_clipping_limit=0.03):

        # Load arguments into class variables
        # todo: add a try except here, which catches non-lists fed in
        # makes them a list of 1
        self.no_of_xs = len(x_lists)
        self.no_of_ys = len(y_lists)

        # Otherwise, zipping a list of length 1 (one patient with many modalities)
        # will zip along the wrong axis and leave unusable paths
        self.one_patient_only = type(x_lists[0]) == str or type(x_lists[0]) == np.str_

        if self.one_patient_only:
            self.x_lists = x_lists
            self.y_lists = y_lists
            #print("One patient loaded: \nx: %s \ny:%s" %(self.x_lists, self.y_lists))
        else:
            self.x_lists = list(zip(*x_lists))[first_vol: first_vol + volumes_to_analyse]
            self.y_lists = list(zip(*y_lists))[first_vol: first_vol + volumes_to_analyse]

        if not self.one_patient_only:
            combined = list(zip(self.x_lists, self.y_lists))
            random.shuffle(combined)
            self.x_lists[:], self.y_lists[:] = zip(*combined)

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.unet = unet

        self.volumes_to_analyse = volumes_to_analyse
        self.validation = validation
        self.secondary_input = secondary_input
        self.spatial_offset = spatial_offset
        self.randomise_spatial_offset = randomise_spatial_offset
        self.reslice_isotropic = reslice_isotropic
        self.x_only = x_only
        self.prediction = prediction
        self.shuffle = shuffle
        self.verbose = verbose
        self.BraTS_like_y = BraTS_like_y
        self.histogram_equalise = histogram_equalise
        self.CLAHE_clipping_limit = CLAHE_clipping_limit

        # Ensures a volume is loaded on first request
        self.x_volume_needed = True
        self.y_volume_needed = True

        # Ensures the same image is not analysed multiple times consecutively
        self.next_x = 0
        self.next_y = 0

        # This ensures correct indexing for patches after the first volume
        self.x_offset = 0
        self.y_offset = 0

        # To benchmark the patch extraction part of the code
        self.time_patching = []

        # Make important calculations, necessary for len method
        self.patches_x = {}
        self.patches_y = {}

        # Gets the number of patches from each volume
        self.total_patches = 0
        self.patches_each_volume = []
        for i in range(self.volumes_to_analyse):
            if self.reslice_isotropic != 0:

                # I have not applied correction here for padding.
                #  Reslice isotropic isn't currently being used and just
                #  here for legacy support

                (patches_i,
                 patches_j,
                 patches_k,
                 patches) = self.get_total_patches(self.reslice_isotropic,
                                                   self.reslice_isotropic,
                                                   self.reslice_isotropic)
            else:

                # Deals with weird errors from single patient, 
                #  multiple modalities
                if self.one_patient_only:
                    image = nib.load(self.x_lists[0])
                else:
                    image = nib.load(self.x_lists[i][0])


                dims = image.header["dim"][1:4]
                # dims = dims - self.spatial_offset

                # Correct for the padding which will be applied later, so
                #  the volume divides into patches exactly
                odd_bits = np.mod(dims, self.patch_size)
                dims_to_pad = [patch_size - odd_bit if odd_bit > 20 else 0
                               for odd_bit in odd_bits]
                dims = dims + dims_to_pad

                #print("estimated dims from header info are: ", dims)

                (patches_i,
                 patches_j,
                 patches_k,
                 patches) = self.get_total_patches(dims[0], dims[1], dims[2])

            self.total_patches += patches
            self.patches_each_volume.append(patches)

        # This helps UnetEvaluator reconstruct the patches into the correct arrangement
        self.patch_arrangement = [patches_i, patches_j, patches_k]
        #print("Estimated patch arrangement from header: ", self.patch_arrangement)

        self.batches_per_volume = patches // int(batch_size)
        self.total_batches = int(np.ceil(self.total_patches / int(batch_size)))

        # Print summary
        if self.verbose:
            print("\n" + "#" * 8 + " New generator intialised with the following "
                                   "properties: " + "#" * 8)
            print("This generator will produce patches from a volume of "
                  "dimensions (after padding): ", dims)
            print("These should correspond to a patch_arrangement of ", self.patch_arrangement)
            print("There should be approximately %i patches for each volume"
                  % patches)
            print("This should lead to approximately %i batches per volume"
                  % self.batches_per_volume)
            print("The total number of batches will be ", self.total_batches)
            print("#" * 8 + " End new generator properties " + "#" * 8 + "\n")

    def __len__(self):
        return self.total_batches

    def get_total_patches(self, dims_i, dims_j, dims_k, only_total=False):
        """Given the 3 dimensions of an image, and the patch size and stride
        already specified in the instance, this will give back how many
        steps will be taken in each of the dimensions"""

        # I have no confidence in my maths, but I've checked this empirically
        patches_i = int(
            np.floor((dims_i - (self.patch_size)) / self.stride) + 1)
        patches_j = int(
            np.floor((dims_j - (self.patch_size)) / self.stride) + 1)
        patches_k = int(
            np.floor((dims_k - (self.patch_size)) / self.stride) + 1)

        total_patches = patches_i * patches_j * patches_k

        if only_total:
            return total_patches
        else:
            return patches_i, patches_j, patches_k, total_patches

    def get_linear_gradients(self, dims):

        grad_i = np.zeros(dims, dtype=np.float)
        grad_j = np.zeros(dims, dtype=np.float)
        grad_k = np.zeros(dims, dtype=np.float)

        for i in range(dims[0]):
            grad_i[i, :, :] = i / dims[0]

        for j in range(dims[1]):
            grad_j[:, j, :] = j / dims[1]

        for k in range(dims[2]):
            grad_k[:, :, k] = k / dims[2]

        linear_gradients = [grad_i, grad_j, grad_k]
        linear_gradients = [self.spatial_offset_func(grad) for grad in linear_gradients]
        linear_gradients = np.moveaxis(linear_gradients, 0, 3)

        return linear_gradients

    def reslice(self, volume):

        if self.reslice_isotropic == 0:
            return volume

        original_dims = np.array(volume.shape)

        # Perform reslicing
        volume = zoom(volume, (self.reslice_isotropic / original_dims))

        return volume

    def spatial_offset_func(self, volume):
        """Crops the volume by self.spatial_offset, then pads it in
        the other direction to preserve dimensions"""

        if self.spatial_offset == 0:
            return volume

        volume = volume[
                 self.spatial_offset:, self.spatial_offset:, self.spatial_offset:]

        volume = np.pad(volume,
                        ((0, self.spatial_offset),
                         (0, self.spatial_offset),
                         (0, self.spatial_offset)),
                        'edge')

        return volume

    def load_volume(self, path):

        volume = nib.load(path).get_data()

        volume = self.spatial_offset_func(volume)
        volume = self.reslice(volume)

        dims = volume.shape
        leeway = 20

        # Calculate dimensions in each axis that are not patched, and pad
        odd_bits = np.mod(dims, self.patch_size)
        # print("Dimensions not patched: ", odd_bits)
        # Doesn't pad when odd bit is 0
        dims_to_pad = [self.patch_size - odd_bit if odd_bit > leeway else 0
                       for odd_bit in odd_bits]

        # Save this so it can be used by e.g. UnetEvaluator to see how much things were padded
        self.dims_to_pad = dims_to_pad

        #print("Dims to pad are ", dims_to_pad)

        # print("Therefore, padding should take dimensions: ", dims_to_pad)
        volume = np.pad(volume,
                        ((dims_to_pad[0] // 2, -(-dims_to_pad[0] // 2)),
                         (dims_to_pad[1] // 2, -(-dims_to_pad[1] // 2)),
                         (dims_to_pad[2] // 2, -(-dims_to_pad[2] // 2))),
                        'minimum')
        dims = volume.shape

        #print("Real dims after padding: ", dims)

        (self.iterations_i,
         self.iterations_j,
         self.iterations_k,
         self.total_indices) = self.get_total_patches(dims[0],
                                                      dims[1],
                                                      dims[2])
        true_patch_arrangement = [self.iterations_i, self.iterations_j, 
                                  self.iterations_k]

        #print("Real patch arrangement: ", true_patch_arrangement)
        total_patches = self.total_indices

        return volume

    def scale(self, volume):

        # Fit a scaler to the current image
        dims = volume.shape
        scaler = StandardScaler()
        volume = volume.reshape(-1, 1)
        volume = scaler.fit_transform(volume)
        volume = volume.reshape(dims[0], dims[1], dims[2])

        return volume

    def create_shuffling_dict(self, seed):

        random.seed(seed)
        # Create a mapping between requested and retrieved index, if shuffling
        requested_indices = list(range(self.total_indices))
        shuffled_indices = random.sample(requested_indices, self.total_indices)
        mapping_dict = {i: shuffled_indices[i] for i in requested_indices}
        reverse_mapping_dict = {shuffled_indices[i]: i
                                for i in requested_indices}

        self.mapping_dict = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        return

    def histogram_equalise_func(self, volume, n_bins=256):

        if not self.histogram_equalise:
            return volume

        squashed_volume = np.reshape(volume, (volume.shape[0], volume.shape[1] * volume.shape[2]))
        img_rescale = exposure.equalize_adapthist(squashed_volume, clip_limit=self.CLAHE_clipping_limit)
        img_rescale = np.reshape(img_rescale, volume.shape)

        return img_rescale

    def patch_from_volume(self, index):
        """Only supports u-nets at the moment. Given an index, it will find
        and return the corresponding single patch for x and y"""

        if self.x_volume_needed:
            if self.randomise_spatial_offset:
                self.spatial_offset = np.random.randint(0, self.patch_size // 2)
                #print("Spatial offset set to ", self.spatial_offset)

            # If only one patient is given, it only has to loop over self.x_lists directly
            # TODO: simplify this by making simgle patient read as a list of 1
            if self.one_patient_only:
                self.x_volumes = [
                    self.load_volume(path) for path in self.x_lists]
            else:
                self.x_volumes = [
                    self.load_volume(path) for path in self.x_lists[self.next_x]]

            self.x_volumes = [self.histogram_equalise_func(volume) for volume in self.x_volumes]
            self.x_volumes = [self.scale(volume) for volume in self.x_volumes]

            # Get linear gradients if secondary input require
            if self.secondary_input:
                self.linear_gradients = self.get_linear_gradients(
                    self.x_volumes[0].shape)

            self.x_volume_needed = False

            if self.shuffle: self.create_shuffling_dict(np.random.seed())

            if self.one_patient_only:
                self.y_volumes = [
                    self.load_volume(path) for path in self.y_lists]
            else:
                self.y_volumes = [
                    self.load_volume(path) for path in self.y_lists[self.next_y]]

            self.y_volume_needed = False

            # Process segmentations with one channel and multiple values in it
            if self.BraTS_like_y:
                volume = self.y_volumes[0]
                self.y_volumes = []
                class_values = [0, 1, 2, 4]
                for i in class_values:
                    # Skip the air class, which is added later anyway
                    if i == 0:
                        continue
                    self.y_volumes.append(volume == i)

        # First, apply the offsets to the index and then shuffle it
        unshuffled_index = index - self.y_offset
        if self.shuffle:
            shuffled_index = self.mapping_dict[unshuffled_index]
        else:
            shuffled_index = unshuffled_index

        # necessary?
        offset = self.y_offset
        vol_dims = self.y_volumes[0].shape

        y_volumes = self.y_volumes

        # necessary?
        offset = self.x_offset
        vol_dims = self.x_volumes[0].shape

        x_volumes = self.x_volumes

        # Figuring out the corresponding patch for the requested index
        ij_area = self.iterations_i * self.iterations_j
        in_plane_index = shuffled_index % ij_area

        # This is the number of steps to take in each direction
        i_no = in_plane_index % self.iterations_i
        j_no = in_plane_index // self.iterations_i
        k_no = shuffled_index // ij_area

        # Convert from which position (i.e. 5th patch) to absolute location
        i = i_no * self.stride
        j = j_no * self.stride
        k = k_no * self.stride

        patcher = Patcher([i, j, k], self.patch_size)

        # Extract the y (label) information
        y_patches = [patcher.patch(volume) for volume in y_volumes]
        y_patches = np.moveaxis(y_patches, 0, 3)

        # Background (air) class
        no_brain = np.ones_like(y_patches[:, :, :, 0]) - np.sum(y_patches, axis=3)
        no_brain = np.expand_dims(no_brain, -1)

        y_patches = np.concatenate([y_patches, no_brain], axis=3)

        # Extract the x (image) information
        x_patches = [patcher.patch(volume) for volume in x_volumes]

        if self.secondary_input:
            secondary_input = patcher.patch(self.linear_gradients)
        else:
            secondary_input = np.zeros([self.patch_size,
                                        self.patch_size,
                                        self.patch_size,
                                        3])

        # Decide whether a new volume has to be loaded next time round
        if self.total_indices - 1 == unshuffled_index:
            # These can become one offset once this is tested and working
            self.y_offset += self.total_indices
            self.x_offset += self.total_indices

            self.y_volume_needed = True
            self.x_volume_needed = True

            self.next_y += 1
            self.next_x += 1

        return x_patches, y_patches, secondary_input

    def __load__(self, index):
        """Returns a single patch as requested by __getitem__. Could probably
        be combined with patch_from_volume, as this currently just calls that.
        However, patch_from_volume is currently unwieldy so that would have
        to be modularised first"""

        x_patches, y_patches, secondary_input = self.patch_from_volume(index)

        # Can try np.expand_dims( ) here instead
        x_patches = [patch_x.reshape(
            self.patch_size, self.patch_size, self.patch_size, 1)
            for patch_x in x_patches]

        # Treats multimodal images as separate channels
        x_patches = np.concatenate(x_patches, axis=3)

        # Combine multimodal / single modality x with the spatial inputs
        x_patches = np.concatenate([x_patches, secondary_input], axis=3)

        return x_patches, y_patches

    def on_epoch_end(self):

        #new_offset = np.random.randint(0, self.patch_size // 2)
        #print("\n" + "#" * 7 + "Epoch ended. Going to randomly change "
        #      "spatial offset to ", new_offset)
        #self.spatial_offset = new_offset

        #print("Going to shuffle the two lists")
        if not self.one_patient_only:
            combined = list(zip(self.x_lists, self.y_lists))
            random.shuffle(combined)
            self.x_lists[:], self.y_lists[:] = zip(*combined)

        self.x_offset = 0
        self.y_offset = 0

        self.next_x = 0
        self.next_y = 0

    def __getitem__(self, batch):

        # print(\n + "#" * 8 + "Requested batch %s from generator" % batch)
        # print("Generator has length ", len(self))

        # print("self.total_patches is %s, and highest index to request is %s"
        #       % (self.total_patches, (batch + 1)*self.batch_size))

        if self.prediction and batch != 0:
            if (batch + 1) * self.batch_size > self.total_patches:
                self.batch_size = self.total_patches - batch * self.batch_size
                # print("Got to the end of the volume - next batch will be smaller")
                # print("Next batch will have size ", self.batch_size)

        batch = [self.__load__(index) for index in
                 range((batch * self.batch_size), (batch + 1) * self.batch_size)]

        batch_x = [data_point[0] for data_point in batch]
        batch_y = [data_point[1] for data_point in batch]

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        x_1 = np.concatenate(batch_x[:, :, :, :, :self.no_of_xs], axis=0)
        x_1 = x_1.reshape(self.batch_size,
                          self.patch_size,
                          self.patch_size,
                          self.patch_size,
                          self.no_of_xs)

        if self.secondary_input:
            x_2 = np.concatenate(batch_x[:, :, :, :, self.no_of_xs:], axis=0)
            x_2 = x_2.reshape(self.batch_size,
                              self.patch_size,
                              self.patch_size,
                              self.patch_size,
                              3)

        if self.x_only:
            if self.secondary_input:
                return [x_1, x_2]
            else:
                return x_1
        else:
            if self.secondary_input:
                return [x_1, x_2], batch_y
            else:
                return x_1, batch_y


class UnetEvaluator(PatchSequence):

    def __init__(self, model, BraTS_like_y=False, batch_size=1,
                 print_every_overlap=False, whole_volume=False, whole_volume_dims=None):

        # Make class variables from arguments
        self.model = model
        self.patch_size = model.layers[0].input_shape[1]
        self.BraTS_like_y = BraTS_like_y
        self.print_every_overlap = print_every_overlap
        # Since this is a unet - for now, I don't want to predict overlapping patches
        self.stride = self.patch_size
        self.secondary_input = True # Removed as a argument, because I don't imagine ever changing it
        self.batch_size = batch_size
        self.whole_volume = whole_volume
        self.whole_volume_dims = whole_volume_dims

        # Class variables that are necessary for inherited methods
        self.next_x = 0
        self.next_y = 0
        self.x_volume_needed = True
        self.y_volume_needed = True

        self.x_offset = 0
        self.y_offset = 0

        self.patches_each_volume = [0]
        self.time_patching = []

        # Placeholders which will be populated later
        self.patch_arrangement = []

    def unpad(self, volume):
        """Removes padding added during generation"""

        vol_dims = volume.shape

        dims_from = [dim // 2 for dim in self.dims_to_pad]
        dims_to = [-dim // 2 for dim in self.dims_to_pad]

        #print("About to unpad from %s to %s" % (dims_from, dims_to))

        volume = volume[dims_from[0]: dims_to[0],
                        dims_from[1]: dims_to[1],
                        dims_from[2]: dims_to[2]]
        
        return volume

    def predict_volume(self, image_path, spatial_offset=0):

        # Calculate number of modalities if not there already
        self.num_modalities = len(image_path)

        generator = PatchSequenceThreaded(
            image_path, image_path,
            batch_size=self.batch_size, patch_size=self.patch_size,
            stride=self.stride, volumes_to_analyse=1,
            secondary_input=self.secondary_input, spatial_offset=spatial_offset,
            x_only=True, prediction=True)#, shuffle=False, verbose=False)

        self.patch_arrangement = generator.patch_arrangement
        self.total_patches = self.patch_arrangement[0] * self.patch_arrangement[1] * self.patch_arrangement[2]

        #print("Going to make predictions on those patches…")
        predicted_volume = self.model.predict_generator(generator, verbose=1)
        predicted_volume_still_in_batches = predicted_volume

        # Also get the dimensions padded, so they can be removed later
        self.dims_to_pad = generator.dims_to_pad

        #print("The dimensions of predicted patches are: ", predicted_volume.shape)

        batches_predicted_on = [generator.__getitem__(batch)[0] for batch in range(len(generator))]
        #print("Len of batchs_predicted_on straight after getting is ", len(batches_predicted_on))
        batches_predicted_on = np.array(np.concatenate(batches_predicted_on, axis=0))
        #print("The dimensions of the patches being predicted on (just after concat): ", batches_predicted_on.shape)

        # If multiple modalities present, display only first
        batches_predicted_on = batches_predicted_on[:, :, :, :, 0]

        batches_predicted_on = np.reshape(batches_predicted_on, (self.total_patches,
                                                                 self.patch_size,
                                                                 self.patch_size,
                                                                 self.patch_size))
        #print("The dimensions of the patches being predicted on are: ", batches_predicted_on.shape)

        # Put together the predicted patches
        predicted_volume = np.reshape(
            predicted_volume, (self.patch_arrangement[2],
                               self.patch_arrangement[1],
                               self.patch_arrangement[0],
                               self.patch_size, self.patch_size, self.patch_size, 4))
        predicted_volume = np.moveaxis(predicted_volume, (0, 1, 2), (4, 2, 0))
        predicted_volume = np.reshape(
            predicted_volume, ((self.patch_arrangement[0] * self.patch_size,
                                self.patch_arrangement[1] * self.patch_size,
                                self.patch_arrangement[2] * self.patch_size, 4)))

        # Put together the raw image patches predicted on
        raw_volume_patched = np.reshape(
            np.asarray(batches_predicted_on), (self.patch_arrangement[2],
                                               self.patch_arrangement[1],
                                               self.patch_arrangement[0],
                                               self.patch_size,
                                               self.patch_size,
                                               self.patch_size))
        # print("Raw volume after first reshaping has shape: ", raw_volume_patched.shape)
        raw_volume_patched = np.moveaxis(raw_volume_patched, (0, 1, 2), (4, 2, 0))
        raw_volume_patched = np.reshape(
            raw_volume_patched, ((self.patch_arrangement[0] * self.patch_size,
                                  self.patch_arrangement[1] * self.patch_size,
                                  self.patch_arrangement[2] * self.patch_size)))


        return predicted_volume, raw_volume_patched, predicted_volume_still_in_batches

    def predict_whole_volume(self, image_path):

        if self.params:
            generator = VolumeSequence(
                [image_path], [image_path],
                reslice_dims=self.whole_volume_dims, 
                batch_size=1, volumes_to_analyse=1,
                histogram_equalise=self.params["histogram_equalise"],
                CLAHE_clipping_limit=self.params["CLAHE_clipping_limit"])
        else:
            generator = VolumeSequence(
                [image_path], [image_path],
                reslice_dims=self.whole_volume_dims, 
                batch_size=1, volumes_to_analyse=1)

        predicted_volume = self.model.predict_generator(generator, verbose=1)

        raw_volume = generator.__getitem__(0)[0]

        predicted_volume = np.mean(predicted_volume, axis=0)
        raw_volume = np.mean(raw_volume, axis=0)
        raw_volume = np.mean(raw_volume, axis=3)

        return predicted_volume, raw_volume

    def predict_overlapping_volumes(self, image_path, overlapping_patches=5,
                                    spatial_offset=1):
        """Call "predict_volume" multiple times with different offsets, to
        avoid the border artefact at the edge of u-net prediction patches"""

        predicted_volumes = []
        raw_volumes_predicted = []

        for i in range(overlapping_patches):
            this_offset = i * spatial_offset
            print("Going to predict with offset %s, which is prediction "
                  "%s out of %s" % (this_offset, i + 1, overlapping_patches))
            # This if statement ensures that only the first (un-offset) raw
            #  volume is visualised, which is in the same space as predictions
            if i == 0:
                (predicted_volume,
                 raw_volumes_predicted,
                 predicted_volume_still_in_batches) = self.predict_volume(
                    image_path, this_offset)
            else:
                predicted_volume, _, _ = self.predict_volume(image_path, this_offset)

            # Get offset images back into a common spatial space
            predicted_volume = np.pad(predicted_volume,
                                      ((this_offset, 0),
                                       (this_offset, 0),
                                       (this_offset, 0),
                                       (0, 0)),
                                      'minimum')
            # print("predicted volume shape after padding is: ", predicted_volume.shape)
            if this_offset != 0:
                predicted_volume = predicted_volume[:-this_offset, :-this_offset, :-this_offset, :]
            # print("predicted volume shape after cutting off end bit: ", predicted_volume.shape)

            predicted_volumes.append(predicted_volume)

        return predicted_volumes, raw_volumes_predicted, predicted_volume_still_in_batches

    def process_predictions(self, predicted_volumes, raw_volumes_predicted):

        if self.repeat_overlap > 1:
            averaged_prediction = np.mean(predicted_volumes, axis = 0)
            #stan_dev_prediction = np.std(predicted_volumes, axis = 0)
            #stan_dev_prediction = np.sum(stan_dev_prediction, axis = 3)
        else:
            averaged_prediction = predicted_volumes

        # Discretising the probabilistic output
        discretised_prediction = np.reshape(averaged_prediction, (-1, 4))
        most_probable_classes = np.argmax(discretised_prediction, axis=1)
        discretised_prediction = np.zeros_like(discretised_prediction)
        discretised_prediction[np.arange(discretised_prediction.shape[0]), most_probable_classes] = 1

        if self.whole_volume:
            discretised_prediction = np.reshape(
                discretised_prediction, 
                ((averaged_prediction.shape[0], 
                  averaged_prediction.shape[1], 
                  averaged_prediction.shape[2], 4)))

        else:
            discretised_prediction = np.reshape(discretised_prediction,
                                      ((self.patch_arrangement[0] * self.patch_size,
                                        self.patch_arrangement[1] * self.patch_size,
                                        self.patch_arrangement[2] * self.patch_size, 4)))

            averaged_prediction = self.unpad(averaged_prediction[0])
            discretised_prediction = self.unpad(discretised_prediction)
            #stan_dev_prediction = self.unpad(stan_dev_prediction)
            raw_volumes_predicted = self.unpad(raw_volumes_predicted)


        return (predicted_volumes, averaged_prediction, discretised_prediction,
                raw_volumes_predicted)

    def eval_volume(self, image_path, seg_path=[0], overlapping_patches=5,
                    spatial_offset=0, repeat_overlap=1, save_avg=False, 
                    show_canonical=False):

        self.repeat_overlap = repeat_overlap

        if spatial_offset == 0:
            spatial_offset = int((self.patch_size / 2) / overlapping_patches)
            if spatial_offset == 0: spatial_offset = 1
            print("Set spatial_offset to %s automatically" % spatial_offset)

        # repeat_overlap produces the overlapping prediction n times to produce subtle
        #  differences in the predictions, for uncertainty inferences
        overlapped_predictions = []
        for i in range(repeat_overlap):
            if repeat_overlap > 1:
                print("Going to produce fully overlapped prediction %s out of %s for "
                      "confidence inference" % (i, repeat_overlap))
            
            # to do: lots of variables being passed back and forward for no reason, just make them class variables
            if self.whole_volume:
                (predicted_volumes, raw_volumes_predicted) = self.predict_whole_volume(image_path)
                # print("Dimensions of predicted_volumes: ", predicted_volumes.shape)
                # print("Dimensions of raw_volumes_predicted: ", raw_volumes_predicted.shape)

            else:
                (predicted_volumes, 
                raw_volumes_predicted, 
                predicted_volume_still_in_batches) = self.predict_overlapping_volumes(
                    image_path, overlapping_patches, spatial_offset)

            (individual_predictions,
            averaged_prediction,
            discretised_prediction,
            raw_volumes_predicted) = self.process_predictions(
                predicted_volumes, raw_volumes_predicted)

            # print("Shape of discretised_prediction", discretised_prediction.shape)

            overlapped_predictions.append(averaged_prediction)

            if repeat_overlap > 1:
                # Produce confidence inference on the fly
                print("overlapped_predictions len before np.std is ", len(overlapped_predictions))
                stan_dev_prediction = np.std(overlapped_predictions, axis=0)
                print("stan_dev_prediction shape right after np.std is", stan_dev_prediction.shape)
                stan_dev_prediction = np.sum(stan_dev_prediction, axis=3)

            # On the fly visualisation
            if self.print_every_overlap:
                visualise_3_axes(stan_dev_prediction)
                visualise_3_axes(averaged_prediction[:, :, :, :3])

        if repeat_overlap > 1:
            # Produce standard deviation from multiple fully overlapped (and therefore
            #  comparable) predicted volumes
            stan_dev_prediction = np.std(overlapped_predictions, axis = 0)
            stan_dev_prediction = np.sum(stan_dev_prediction, axis = 3)

        # Load the segmentations and combine them. Assumes that if no
        #  c1 (white) segmentation given, that none were
        if len(seg_path) == 3:
            # print("Segmentations present")
            seg_path, seg_path_2, seg_path_3 = seg_path[0], seg_path[1], seg_path[2]
            segs_present = True
        else:
            # print("Segmentations not present")
            segs_present = False

        # TODO: make this work with arbitrary number of segmentations
        if seg_path != 0 and segs_present:
            y_data = nib.load(seg_path).get_data()
            y_data_2 = nib.load(seg_path_2).get_data()
            y_data_3 = nib.load(seg_path_3).get_data()
            SPM_volume = [y_data, y_data_2, y_data_3]
            SPM_volume = np.array(SPM_volume)
            SPM_volume = np.moveaxis(SPM_volume, 0, 3)
            SPM_shape = SPM_volume.shape
            # print("Dimensions of the combined segmentations (from file) is: ", SPM_volume.shape)
        else:
            # print("No segmentations provided (at least for seg_path) so "
            #       "segmentation will be set to zero")
            SPM_volume = np.zeros_like(raw_volumes_predicted)
            SPM_shape = SPM_volume.shape

        ########## Debugging code ##########
        # print("Dimensions of raw_volumes_predicted is: ", raw_volumes_predicted.shape)
        # print("Dimensions of averaged_prediction is: ", averaged_prediction.shape)
        # print("Dimensions of discretised_prediction is: ", discretised_prediction.shape)

        ########## Display the images ##########
        visualise_vol_list([raw_volumes_predicted,
                            averaged_prediction[:, :, :, :3],
                            # stan_dev_prediction,
                            # discretised_prediction[:, :, :, :3],
                            SPM_volume])

        # Display nice canonical images for e.g. poster presentations
        if show_canonical:
            visualise_canonical(raw_volumes_predicted)
            visualise_canonical(averaged_prediction[:, :, :, :3])
            visualise_canonical(SPM_volume)

        # Saves the predictions in the same directory as the raw image
        if save_avg:
            # The below might required image_path to be a list, even a list of 1
            sample_nifti = nib.load(image_path[0])
            pred_nifti = nib.Nifti1Image(averaged_prediction[:, :, :, :],
                                         sample_nifti.affine, sample_nifti.header)
            current_filename = image_path[0].split("/")[-1]
            current_file_dir = image_path[0].replace(current_filename, "")

            new_filename = current_file_dir + (current_filename.split(".")[0] + "_" +
                           socket.gethostname() + "GM_WM_CSF_pred_" +
                           str(np.random.randint(0, 999999)).zfill(6)) + ".nii.gz"

            nib.save(pred_nifti, new_filename)

        return averaged_prediction, discretised_prediction, SPM_volume

    def eval_nifti_categoriser(self, nifti_categoriser, params=None, limit=None):

        # build_train_evaluate can pass in parameter dictionary,
        #  very useful for e.g. histogram equalisation
        self.params = params

        for i in range(len(nifti_categoriser.raw)):

            self.eval_volume(
                [nifti_categoriser.raw[i]], 
                [nifti_categoriser.seg_1[i],
                 nifti_categoriser.seg_2[i],
                 nifti_categoriser.seg_3[i]], overlapping_patches=1,
                    spatial_offset=1, repeat_overlap=1)

            if limit and limit == i + 1: break

        return
