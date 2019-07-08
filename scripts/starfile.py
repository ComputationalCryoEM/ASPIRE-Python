from aspyre.utils.config import ConfigArgumentParser
from aspyre.source.star import Starfile
from aspyre.basis.fb_3d import FBBasis3D
from aspyre.estimation.mean import MeanEstimator
from aspyre.estimation.covar import CovarianceEstimator
from aspyre.estimation.noise import WhiteNoiseEstimator


if __name__ == '__main__':

    parser = ConfigArgumentParser(description='Estimate mean volume and covariance from a starfile.')
    parser.add_argument('--starfile', required=True)
    parser.add_argument('--pixel_size', default=1, type=float)
    parser.add_argument('--ignore_missing_files', action='store_true')
    parser.add_argument('--max_rows', default=None, type=int)
    parser.add_argument('-L', default=16, type=int)

    with parser.parse_args() as args:

        source = Starfile(
            args.starfile,
            pixel_size=args.pixel_size,
            ignore_missing_files=args.ignore_missing_files,
            max_rows=args.max_rows
        )

        L = args.L
        source.set_max_resolution(L)
        source.cache()

        source.whiten()
        basis = FBBasis3D((L, L, L))
        mean_estimator = MeanEstimator(source, basis, batch_size=8192)
        mean_est = mean_estimator.estimate()

        noise_estimator = WhiteNoiseEstimator(source, batchSize=500)
        # Estimate the noise variance. This is needed for the covariance estimation step below.
        noise_variance = noise_estimator.estimate()
        print(f'Noise Variance = {noise_variance}')

        # Passing in a mean_kernel argument to the following constructor speeds up some calculations
        covar_estimator = CovarianceEstimator(source, basis, mean_kernel=mean_estimator.kernel)
        covar_estimator.estimate(mean_est, noise_variance)
