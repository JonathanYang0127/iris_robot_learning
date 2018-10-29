"""
Skew the dataset so that it turns into generating a uniform distribution.
"""
import json
import sys

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skvideo.io import vwrite
from torch import nn as nn
from torch.optim import Adam

import railrl.pythonplusplus as ppp
import railrl.torch.vae.skew.skewed_vae as sv
from railrl.core import logger
from railrl.misc import visualization_util as vu
from railrl.misc.html_report import HTMLReport
from railrl.misc.visualization_util import gif
from railrl.torch.vae.skew.common import (
    Dynamics, plot_curves,
    visualize_samples,
)
from railrl.torch.vae.skew.datasets import project_samples_square_np
from railrl.torch.vae.skew.histogram import Histogram

K = 6

"""
Plotting
"""


def visualize_vae_samples(
        epoch, training_data, vae,
        report, dynamics,
        n_vis=1000,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5)
):
    plt.figure()
    plt.suptitle("Epoch {}".format(epoch))
    n_samples = len(training_data)
    skip_factor = max(n_samples // n_vis, 1)
    training_data = training_data[::skip_factor]
    reconstructed_samples = vae.reconstruct(training_data)
    generated_samples = vae.sample(n_vis)
    projected_generated_samples = dynamics(generated_samples)
    plt.subplot(2, 2, 1)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Generated Samples")
    plt.subplot(2, 2, 2)
    plt.plot(projected_generated_samples[:, 0],
             projected_generated_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Projected Generated Samples")
    plt.subplot(2, 2, 3)
    plt.plot(training_data[:, 0], training_data[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Training Data")
    plt.subplot(2, 2, 4)
    plt.plot(reconstructed_samples[:, 0], reconstructed_samples[:, 1], '.')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.title("Reconstruction")

    fig = plt.gcf()
    sample_img = vu.save_image(fig)
    report.add_image(sample_img, "Epoch {} Samples".format(epoch))

    return sample_img


def visualize_vae(vae, skew_config, report,
                  resolution=20,
                  title="VAE Heatmap"):
    xlim, ylim = vae.get_plot_ranges()
    show_prob_heatmap(vae, xlim=xlim, ylim=ylim, resolution=resolution)
    fig = plt.gcf()
    prob_heatmap_img = vu.save_image(fig)
    report.add_image(prob_heatmap_img, "Prob " + title)

    show_weight_heatmap(
        vae, skew_config, xlim=xlim, ylim=ylim, resolution=resolution,
    )
    fig = plt.gcf()
    heatmap_img = vu.save_image(fig)
    report.add_image(heatmap_img, "Weight " + title)
    return prob_heatmap_img


def show_weight_heatmap(
        vae, skew_config,
        xlim, ylim,
        resolution=20,
):

    def get_prob_batch(batch):
        prob = vae.compute_density(batch)
        return prob_to_weight(prob, skew_config)

    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim,
                                resolution=resolution, batch=True)
    vu.plot_heatmap(heat_map)


def show_prob_heatmap(
        vae,
        xlim, ylim,
        resolution=20,
):

    def get_prob_batch(batch):
        return vae.compute_density(batch)

    heat_map = vu.make_heat_map(get_prob_batch, xlim, ylim,
                                resolution=resolution, batch=True)
    vu.plot_heatmap(heat_map)


def progressbar(it, prefix="", size=60):
    count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        sys.stdout.write(
            "%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), _i, count))
        sys.stdout.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i + 1)
    sys.stdout.write("\n")
    sys.stdout.flush()


def train_from_variant(variant):
    variant.pop('seed')
    variant.pop('exp_id')
    variant.pop('exp_prefix')
    variant.pop('unique_id')
    variant.pop('instance_type')
    train(full_variant=variant, **variant)


def prob_to_weight(prob, skew_config):
    weight_type = skew_config['weight_type']
    min_prob = skew_config['minimum_prob']
    if min_prob:
        prob = np.maximum(prob, min_prob)
    with np.errstate(divide='ignore', invalid='ignore'):
        if weight_type == 'inv_p':
            weights = 1. / prob
        elif weight_type == 'nll':
            weights = - np.log(prob)
        elif weight_type == 'sqrt_inv_p':
            weights = (1. / prob) ** 0.5
        elif weight_type == 'exp':
            exp = skew_config['alpha']
            weights = prob ** exp
        else:
            raise NotImplementedError()
    weights[weights == np.inf] = 0
    weights[weights == -np.inf] = 0
    weights[weights == -np.nan] = 0
    return weights / weights.flatten().sum()


def train(
        dataset_generator,
        n_start_samples,
        projection=project_samples_square_np,
        n_samples_to_add_per_epoch=1000,
        n_epochs=100,
        z_dim=1,
        hidden_size=32,
        save_period=10,
        append_all_data=True,
        full_variant=None,
        dynamics_noise=0,
        decoder_output_var='learned',
        num_bins=5,
        skew_config=None,
        use_perfect_samples=False,
        use_perfect_density=False,
        reset_vae_every_epoch=False,
        vae_kwargs=None,
        use_dataset_generator_first_epoch=True,
):

    """
    Sanitize Inputs
    """
    assert skew_config is not None
    if not (use_perfect_density and use_perfect_samples):
        assert vae_kwargs is not None
    if vae_kwargs is None:
        vae_kwargs = {}

    report = HTMLReport(
        logger.get_snapshot_dir() + '/report.html',
        images_per_row=10,
    )
    dynamics = Dynamics(projection, dynamics_noise)
    if full_variant:
        report.add_header("Variant")
        report.add_text(
            json.dumps(
                ppp.dict_to_safe_json(
                    full_variant,
                    sort=True),
                indent=2,
            )
        )

    vae, decoder, decoder_opt, encoder, encoder_opt = get_vae(
        decoder_output_var,
        hidden_size,
        z_dim,
        vae_kwargs,
    )

    epochs = []
    losses = []
    kls = []
    log_probs = []
    hist_heatmap_imgs = []
    vae_heatmap_imgs = []
    sample_imgs = []
    entropies = []
    tvs_to_uniform = []
    entropy_gains_from_reweighting = []
    """
    p_theta = VAE's distribution
    """
    p_theta = Histogram(num_bins)
    p_new = Histogram(num_bins)

    orig_train_data = dataset_generator(n_start_samples)
    train_data = orig_train_data
    for epoch in progressbar(range(n_epochs)):
        p_theta = Histogram(num_bins)
        if epoch == 0 and use_dataset_generator_first_epoch:
            vae_samples = dataset_generator(n_samples_to_add_per_epoch)
        else:
            if use_perfect_samples and epoch != 0:
                # Ideally the VAE = p_new, but in practice, it won't be...
                vae_samples = p_new.sample(n_samples_to_add_per_epoch)
            else:
                vae_samples = vae.sample(n_samples_to_add_per_epoch)
        projected_samples = dynamics(vae_samples)
        if append_all_data:
            train_data = np.vstack((train_data, projected_samples))
        else:
            train_data = np.vstack((orig_train_data, projected_samples))

        p_theta.fit(train_data)
        if use_perfect_density:
            prob = p_theta.compute_density(train_data)
        else:
            prob = vae.compute_density(train_data)
        all_weights = prob_to_weight(prob, skew_config)
        p_new.fit(train_data, weights=all_weights)
        if epoch == 0 or (epoch + 1) % save_period == 0:
            epochs.append(epoch)
            report.add_text("Epoch {}".format(epoch))
            hist_heatmap_img = visualize_histogram(p_theta, skew_config, report)
            vae_heatmap_img = visualize_vae(
                vae, skew_config, report,
                resolution=num_bins,
            )
            sample_img = visualize_vae_samples(
                epoch, train_data, vae, report, dynamics,
            )

            visualize_samples(
                p_theta.sample(n_samples_to_add_per_epoch),
                report,
                title="P Theta/RB Samples",
            )
            visualize_samples(
                p_new.sample(n_samples_to_add_per_epoch),
                report,
                title="P Adjusted Samples",
            )
            hist_heatmap_imgs.append(hist_heatmap_img)
            vae_heatmap_imgs.append(vae_heatmap_img)
            sample_imgs.append(sample_img)
            report.save()

            Image.fromarray(hist_heatmap_img).save(
                logger.get_snapshot_dir() + '/hist_heatmap{}.png'.format(epoch)
            )
            Image.fromarray(vae_heatmap_img).save(
                logger.get_snapshot_dir() + '/hist_heatmap{}.png'.format(epoch)
            )
            Image.fromarray(sample_img).save(
                logger.get_snapshot_dir() + '/samples{}.png'.format(epoch)
            )

        """
        train VAE to look like p_new
        """
        if sum(all_weights) == 0:
            all_weights[:] = 1
        if reset_vae_every_epoch:
            vae, decoder, decoder_opt, encoder, encoder_opt = get_vae(
                decoder_output_var,
                hidden_size,
                z_dim,
                vae_kwargs,
            )
        vae.fit(train_data, weights=all_weights)
        epoch_stats = vae.get_epoch_stats()

        losses.append(np.mean(epoch_stats['losses']))
        kls.append(np.mean(epoch_stats['kls']))
        log_probs.append(np.mean(epoch_stats['log_probs']))
        entropies.append(p_theta.entropy())
        tvs_to_uniform.append(p_theta.tv_to_uniform())
        entropy_gain = p_new.entropy() - p_theta.entropy()
        entropy_gains_from_reweighting.append(entropy_gain)

        logger.record_tabular("Epoch", epoch)
        logger.record_tabular("VAE Loss", np.mean(epoch_stats['losses']))
        logger.record_tabular("VAE KL", np.mean(epoch_stats['kls']))
        logger.record_tabular("VAE Log Prob", np.mean(epoch_stats['log_probs']))
        logger.record_tabular('Entropy ', p_theta.entropy())
        logger.record_tabular('KL from uniform', p_theta.kl_from_uniform())
        logger.record_tabular('TV to uniform', p_theta.tv_to_uniform())
        logger.record_tabular('Entropy gain from reweight', entropy_gain)
        logger.dump_tabular()
        logger.save_itr_params(epoch, {
            'vae': vae,
            'train_data': train_data,
            'vae_samples': vae_samples,
        })

    report.add_header("Training Curves")
    plot_curves(
        [
            ("Training Loss", losses),
            ("KL", kls),
            ("Log Probs", log_probs),
            ("Entropy Gain from Reweighting", entropy_gains_from_reweighting),
        ],
        report,
    )
    plot_curves(
        [
            ("Entropy", entropies),
            ("TV to Uniform", tvs_to_uniform),
        ],
        report,
    )
    report.add_text("Max entropy: {}".format(p_theta.max_entropy()))
    report.save()

    for filename, imgs in [
        ("hist_heatmaps", hist_heatmap_imgs),
        ("vae_heatmaps", vae_heatmap_imgs),
        ("samples", sample_imgs),
    ]:
        video = np.stack(imgs)
        vwrite(
            logger.get_snapshot_dir() + '/{}.mp4'.format(filename),
            video,
        )
        gif_file_path = logger.get_snapshot_dir() + '/{}.gif'.format(filename)
        gif(gif_file_path, video)
        report.add_image(gif_file_path, txt=filename, is_url=True)
    report.save()


def get_vae(decoder_output_var, hidden_size, z_dim, vae_kwargs):
    encoder = sv.Encoder(
        nn.Linear(2, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, z_dim * 2),
    )
    if decoder_output_var == 'learned':
        last_layer = nn.Linear(hidden_size, 4)
    else:
        last_layer = nn.Linear(hidden_size, 2)
    decoder = sv.Decoder(
        nn.Linear(z_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        last_layer,
        output_var=decoder_output_var,
        output_offset=-1,
    )
    encoder_opt = Adam(encoder.parameters())
    decoder_opt = Adam(decoder.parameters())
    vae = sv.VAE(encoder=encoder, decoder=decoder, z_dim=z_dim, **vae_kwargs)
    return vae, decoder, decoder_opt, encoder, encoder_opt


def visualize_histogram(histogram, skew_config, report, title=""):
    prob = histogram.pvals
    weights = prob_to_weight(prob, skew_config)
    xrange, yrange = histogram.xy_range
    extent = [xrange[0], xrange[1], yrange[0], yrange[1]]
    for name, values in [
        ('Weight Heatmap', weights),
        ('Prob Heatmap', prob),
    ]:
        plt.figure()
        fig = plt.gcf()
        ax = plt.gca()
        values = values.copy()
        values[values == 0] = np.nan
        heatmap_img = ax.imshow(
            np.swapaxes(values, 0, 1),  # imshow uses first axis as y-axis
            extent=extent,
            cmap=plt.get_cmap('plasma'),
            interpolation='nearest',
            aspect='auto',
            origin='bottom',  # <-- Important! By default top left is (0, 0)
            # norm=LogNorm(),
        )
        divider = make_axes_locatable(ax)
        legend_axis = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(heatmap_img, cax=legend_axis, orientation='vertical')
        heatmap_img = vu.save_image(fig)
        if histogram.num_bins < 5:
            pvals_str = np.array2string(histogram.pvals, precision=3)
            report.add_text(pvals_str)
        report.add_image(heatmap_img, "{} {}".format(title, name))
    return heatmap_img