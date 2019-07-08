import numpy as np
from aspyre.utils.numeric import xp


class PickerHelper:

    @classmethod
    def gaussian_filter(cls, size_filter, std):
        """Computes low-pass filter.
        
        Args:
            size_filter: Size of filter (size_filter x size_filter).
            std: sigma value in filter.
        """

        y, x = xp.mgrid[-(size_filter - 1) // 2: (size_filter - 1) // 2 + 1,
                        -(size_filter - 1) // 2: (size_filter - 1) // 2 + 1]

        response = xp.exp(-xp.square(x) - xp.square(y) / (2*(std**2)))/(xp.sqrt(2*xp.pi)*std)
        response[response < xp.finfo('float').eps] = 0

        return xp.asnumpy(response / response.sum())  # Normalize so sum is 1

    @classmethod
    def extract_windows(cls, img, block_size):
        """Extracts blocks of size (block_size x block_size) from the micrograph. Blocks are 
        extracted with steps of size (block_size)
        
        Args:
            img: Micrograph image.
            block_size: required block size.
            
        Returns:
            3D Matrix of blocks. For example, img[0] is the first block.
        """

        # keep only the portion of the image that can be split into blocks with no remainder
        img = xp.asarray(img[:-(img.shape[0] % block_size), :-(img.shape[1] % block_size)])

        dim3_size = np.sqrt(np.prod(img.shape) // (block_size ** 2)).astype(int)

        img = xp.reshape(img, (block_size, dim3_size, block_size, dim3_size), 'F')

        img = xp.transpose(img, (0, 2, 1, 3))
        img = xp.reshape(img, (img.shape[0] * img.shape[1], img.shape[2] * img.shape[3]), 'F')

        return img

    @classmethod
    def extract_query(cls, img, block_size):
        """Extract all query images from the micrograph. windows are 
        extracted with steps of size (block_size/2)
        
        Args:
            img: Micrograph image.
            block_size: Query images must be of size (block_size x block_size).
            
        Returns:
            4D Matrix of query images. 
        """

        # keep only the portion of the image that can be split into blocks with no remainder
        blocks = xp.asarray(img[:-(img.shape[0] % block_size), :-(img.shape[1] % block_size)])

        dim3_size = np.sqrt(np.prod(blocks.shape) // (block_size ** 2)).astype(int)
        blocks = xp.reshape(blocks, (block_size, dim3_size, block_size,  dim3_size), 'F')

        blocks = xp.transpose(blocks, (0, 2, 1, 3))

        blocks = xp.reshape(blocks, (blocks.shape[0], blocks.shape[1], -1), 'F')

        blocks = xp.concatenate(
            (blocks,
             xp.concatenate((blocks[:, :, 1:],
                             xp.reshape(blocks[:, :, 0], (blocks.shape[0], blocks.shape[1], 1), 'F')),
                            axis=2)), axis=0)

        temp = xp.concatenate((blocks[:, :, int(np.floor(2 * img.shape[1] / 2 / block_size)):],
                               blocks[:, :, 0:int(np.floor(2 * img.shape[1] / 2 / block_size))]),
                              axis=2)

        blocks = xp.concatenate((blocks, temp), axis=1)

        blocks = xp.reshape(blocks,
                            (2 * block_size, 2 * block_size,
                             int(np.floor(2 * img.shape[0] / 2 / block_size)),
                             int(np.floor(2 * img.shape[1] / 2 / block_size))), 'F')

        blocks = blocks[:, :, 0:blocks.shape[2]-1, 0:blocks.shape[3]-1]

        blocks = xp.transpose(blocks, (2, 3, 0, 1))

        return blocks

    @classmethod
    def reference_size(cls, img, container_size):
        num_containers_row = img.shape[0] // container_size
        num_containers_col = img.shape[1] // container_size

        return num_containers_row * num_containers_col * 4

    @classmethod
    def extract_references(cls, img, query_size, container_size):
        """Chooses and extracts reference images from the micrograph. 
        
        Args:
            img: Micrograph image.
            query_size: Reference images must be of the same size of query images, i.e. (query_size x query_size).
            container_size: Containers are large regions used to select reference images. The size of each 
            region is (container_size x container_size)
            
        Returns:
            3D Matrix of reference images.  windows[0] is the first reference window.
        """

        img = xp.asarray(img)
        num_containers_row = img.shape[0] // container_size
        num_containers_col = img.shape[1] // container_size

        windows = xp.zeros((cls.reference_size(img, container_size), query_size, query_size))
        win_idx = 0

        mean_all, std_all = cls.moments(img, query_size)

        for y_contain in range(num_containers_row):
            for x_contain in range(num_containers_col):
                temp = img[
                    y_contain * container_size: min(img.shape[0], (y_contain+1) * container_size),
                    x_contain * container_size: min(img.shape[1], (x_contain+1) * container_size)
                ]

                y_start = y_contain * container_size + query_size - 1
                y_end = min(mean_all.shape[0] - query_size, (y_contain + 1) * container_size)

                x_start = x_contain * container_size + query_size - 1
                x_end = min(mean_all.shape[1] - query_size, (x_contain + 1) * container_size)

                mean_contain = mean_all[y_start: y_end, x_start: x_end]
                std_contain = std_all[y_start: y_end, x_start: x_end]

                ind = xp.argmax(mean_contain)
                if ind.size == 1:
                    y, x = xp.unravel_index(ind, mean_contain.shape)
                    windows[win_idx, :, :] = temp[y: y + query_size, x: x + query_size]
                    win_idx += 1

                ind = xp.argmin(mean_contain)
                if ind.size == 1:
                    y, x = xp.unravel_index(ind, mean_contain.shape)
                    windows[win_idx, :, :] = temp[y: y + query_size, x: x + query_size]
                    win_idx += 1

                ind = xp.argmax(std_contain)
                if ind.size == 1:
                    y, x = xp.unravel_index(ind, std_contain.shape)
                    windows[win_idx, :, :] = temp[y: y + query_size, x: x + query_size]
                    win_idx += 1
                    
                ind = xp.argmin(std_contain)
                if ind.size == 1:
                    y, x = xp.unravel_index(ind, std_contain.shape)
                    windows[win_idx, :, :] = temp[y: y + query_size, x: x + query_size]
                    win_idx += 1

        return windows

    @classmethod
    def get_training_set(cls, micro_img, bw_mask_p, bw_mask_n, n):
        """Gets training set for the SVM classifier.
        
        Args:
            micro_img: Micrograph image.
            bw_mask_p: Binary image indicating regions from which to extract examples of particles.
            bw_mask_n: Binary image indicating regions from which to extract examples of noise.
            n: Size of training windows.
            
        Returns:
            A matrix of features and a vector of labels for the SVM training.
        """

        non_overlap = cls.extract_windows(micro_img, n)

        indicate = cls.extract_windows(bw_mask_p, n)
        # consider only windows with no noise
        windows = non_overlap[:, (indicate != 0).all(axis=0)]
        p_mu = xp.mean(windows, axis=0)
        p_std = xp.std(windows, axis=0)

        indicate = cls.extract_windows(bw_mask_n, n)
        # consider only windows with no particles
        windows = non_overlap[:, (indicate != 1).all(axis=0)]
        n_mu = xp.mean(windows, axis=0)
        n_std = xp.std(windows, axis=0)

        p_mu = xp.reshape(p_mu, (p_mu.shape[0], 1), 'F')
        p_std = xp.reshape(p_std, (p_std.shape[0], 1), 'F')
        n_mu = xp.reshape(n_mu, (n_mu.shape[0], 1), 'F')
        n_std = xp.reshape(n_std, (n_std.shape[0], 1), 'F')

        x = xp.concatenate((p_mu, p_std), axis=1)
        x = xp.concatenate((x, xp.concatenate((n_mu, n_std), axis=1)), axis=0)

        y = xp.concatenate((xp.ones(p_mu.shape[0]), xp.zeros(n_mu.shape[0])), axis=0)

        return x, y

    @classmethod
    def moments(cls, img, query_size):
        """Calculates the mean and standard deviation for each window of size (query_size x query_size) 
        in the micrograph.
        
        Args:
            img: Micrograph image.
            query_size: Size of windows for which to compute mean and std.
            
        Returns:
            A matrix of mean intensity and a matrix of variance, each containing a single 
            entry for each possible (query_size x query_size) window in the micrograph.
        """
        
        filt = xp.ones((query_size, query_size)) / (query_size * query_size)
        filt = xp.pad(filt, (0, img.shape[0]-1), 'constant', constant_values=(0, 0))
        filt_freq = xp.fft2(filt, axes=(0, 1))

        pad_img = xp.pad(img, (0, query_size - 1), 'constant', constant_values=(0, 0))
        img_freq = xp.fft2(pad_img, axes=(0, 1))

        mean_freq = xp.multiply(img_freq, filt_freq)
        mean_all = xp.ifft2(mean_freq, axes=(0, 1)).real

        pad_img_square = np.square(pad_img)
        img_var_freq = xp.fft2(pad_img_square, axes=(0, 1))
        var_freq = xp.multiply(img_var_freq, filt_freq)
        var_all = xp.ifft2(var_freq, axes=(0, 1))
        var_all = var_all.real - xp.power(mean_all, 2)
        std_all = xp.sqrt(xp.maximum(0, var_all))

        return mean_all, std_all
