import os
from typing import Tuple, List

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, kstest, ranksums, ttest_ind
from scipy.interpolate import UnivariateSpline
import copy
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
from common import PhenotypeFramework as pf
from revolve2.core.optimization import DbId
from revolve2.genotypes.cppnwin.modular_robot.body_genotype_v1 import develop_v1
from revolve2.core.modular_robot import Body, Brick, ActiveHinge
from revolve2.core.modular_robot import MorphologicalMeasures
from ._load_db import load_db_novelty, load_db_fitness

class EAPlots:
    plt.rcParams['font.size'] = 18

    @classmethod
    def plot_avg_shape(cls, genotypes: List, size: int = 28, save_as:str=None) -> None:
        assert int(size / 2) == size // 2, "ERROR: size must be a divisible by 2"
        images = [cls._get_img(develop_v1(pf.deserialize(genotype)), size) for genotype in genotypes]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        im_xy, im_xz, im_yz = None, None, None
        for img_xy, img_xz, img_yz in images:
            im_xy = im_xy + img_xy if im_xy is not None else img_xy
            im_xz = im_xz + img_xz if im_xz is not None else img_xz
            im_yz = im_yz + img_yz if im_yz is not None else img_yz

        masked_xy = np.ma.masked_where(im_xy == 0, im_xy)
        masked_xz = np.ma.masked_where(im_xz == 0, im_xz)
        masked_yz = np.ma.masked_where(im_yz == 0, im_yz)

        mmin = np.amin([np.amin(im_xy), np.amin(im_xz), np.amin(im_yz)])
        mmax = np.amax([np.amax(im_xy), np.amax(im_xz), np.amax(im_yz)])

        cmap = copy.copy(plt.cm.get_cmap("inferno"))
        cmap.set_bad(alpha=0)

        ax1.imshow(masked_xy, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax1.set_xlabel("y")
        ax1.set_ylabel("x")

        ax2.imshow(masked_xz, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax2.set_xlabel("z")
        ax2.set_ylabel("x")

        lim = ax3.imshow(masked_yz, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax3.set_xlabel("z")
        ax3.set_ylabel("y")

        cbar_ax = fig.add_axes([0.91, 0.155, 0.01, 0.68])
        fig.colorbar(lim, cax=cbar_ax)

        for ax in [ax1, ax2, ax3]:
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        if save_as is not None:
            plt.savefig(f"plots/{save_as}.png", transparent=True)


    @classmethod
    def plot_avg_shape_aggregate(cls, databases: List[str], db_id: DbId = DbId("optmodular"), size: int = 28, norm: str = None, save_as:str=None) -> None:
        assert int(size / 2) == size // 2, "ERROR: size must be a divisible by 2"
        assert len(databases) > 0, "ERROR: empty databases array"
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        im_xy, im_xz, im_yz = None, None, None
        for database in databases:
            df = load_db_novelty(database, db_id)
            genotypes = df["serialized_multineat_genome"].loc[
                df["generation_index"] == df["generation_index"].max()].tolist()

            images = [cls._get_img(develop_v1(pf.deserialize(genotype)), size) for genotype in genotypes]
            del genotypes
            for img_xy, img_xz, img_yz in images:
                im_xy = im_xy + img_xy if im_xy is not None else img_xy
                im_xz = im_xz + img_xz if im_xz is not None else img_xz
                im_yz = im_yz + img_yz if im_yz is not None else img_yz
            del images

        if norm == "log":
            im_xy = np.ma.log(im_xy).filled(0)
            im_xz = np.ma.log(im_xz).filled(0)
            im_yz = np.ma.log(im_yz).filled(0)

        masked_xy = np.ma.masked_where(im_xy == 0, im_xy)
        masked_xz = np.ma.masked_where(im_xz == 0, im_xz)
        masked_yz = np.ma.masked_where(im_yz == 0, im_yz)


        mmin = np.amin([np.amin(im_xy), np.amin(im_xz), np.amin(im_yz)])
        mmax = np.amax([np.amax(im_xy), np.amax(im_xz), np.amax(im_yz)])

        cmap = copy.copy(plt.cm.get_cmap("inferno"))
        cmap.set_bad(alpha=0)

        ax1.imshow(masked_xy, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax1.set_xlabel("y")
        ax1.set_ylabel("x")

        ax2.imshow(masked_xz, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax2.set_xlabel("z")
        ax2.set_ylabel("x")

        lim = ax3.imshow(masked_yz, origin='lower', cmap=cmap, vmin=mmin, vmax=mmax)
        ax3.set_xlabel("z")
        ax3.set_ylabel("y")

        cbar_ax = fig.add_axes([0.91, 0.155, 0.01, 0.68])
        fig.colorbar(lim, cax=cbar_ax)

        for ax in [ax1, ax2, ax3]:
            ax.hlines(y=size // 2 - 0.5, xmin=0, xmax=size, alpha=0.5)
            ax.vlines(x=size // 2 - 0.5, ymin=0, ymax=size, alpha=0.5)
            ax.set_ylim(0, size)
            ax.set_xlim(0, size)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        if save_as is not None:
            plt.savefig(f"plots/{save_as}.png", transparent=True)
        plt.show()

    @classmethod
    def plot_novelty_fintess_averages(cls, n, f, f_m, save_as:str=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ticks = n.columns.tolist()
        ticks[-1] = "Prod"

        sns.boxplot(ax=ax1, data=n.iloc[0], color="blue")
        ax1.set_xticklabels(ticks)
        ax1.set_ylabel(f"Novelty Average")
        ax1.set_xlabel("Novelty Configuration")
        ax1.set(ylim=(0, 1))

        sns.boxplot(ax=ax2, data=f.iloc[0], color="red")
        ax2.set_xticklabels(ticks)
        ax2.set_ylabel(f"Fitness Mean")
        ax2.set_xlabel("Novelty Configuration")
        ax2.set(ylim=(0, None))

        sns.boxplot(ax=ax3, data=f_m.iloc[0], color="red")
        ax3.set_xticklabels(ticks)
        ax3.set_ylabel(f"Fitness Max")
        ax3.set_xlabel("Novelty Configuration")
        ax3.set(ylim=(None, None))

        plt.tight_layout()
        if save_as is not None:
            cfig = plt.gcf()
            cfig.savefig(f"plots/{save_as}.png", transparent=True)
        plt.show()

    @classmethod
    def fit_novelty_fitness_averages(cls, novelty, fitness_mean, fitness_max, save_as:str=None):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        yss = []
        none_yss = []

        def _find_roots(x, y):
            s = np.abs(np.diff(np.sign(y))).astype(bool)
            return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)

        def _plot_fitted(ax, data, y_axs:str):
            xs, ys = [], []
            my, mx = [], []
            for col, val in zip(data.columns, data.iloc[0, :].values):
                lv = len(val)
                if "None" not in col:
                    mx.append(float(col))
                    my.append(sum(val)/lv)

                    cols = [float(col)] * lv
                    xs.extend(cols)
                    ys.extend(val)
                    ax.scatter(cols, val, color="grey", alpha=0.5, s=10)

            s = UnivariateSpline(x=mx, y=my, s=1)
            x_new = np.linspace(xs[0], xs[-1], 100)
            y_new = s(x_new)
            ax.plot(x_new, y_new, label="Fitted Curve")

            none_vals = data["None"].values[0]
            none_mean = sum(none_vals) / len(none_vals)

            atol = 0.5
            while True:
                n_w = x_new[np.isclose(none_mean, y_new,atol=atol)]
                if len(n_w) > 1:
                    atol *= 0.95
                    none_where = n_w
                elif len(n_w) == 1:
                    none_where=n_w[0]
                    break
                else:
                    none_where = none_where[-1]
                    break


            ax.plot(mx, my, color="red", alpha=0.5)
            ax.scatter([none_where] * len(none_vals), none_vals, label=f"Prod, x={none_where:.3f}", color="green", alpha=0.5, s=10)

            ticks = list(set(xs))
            ax.set_xticks(ticks)
            ax.set_ylabel(y_axs)
            ax.set_xlabel("Novelty Config")
            ax.legend(fontsize=16)
            yss.append(ys)
            none_yss.append(none_vals)

        _plot_fitted(ax1, novelty, "Novelty")
        _plot_fitted(ax2, fitness_mean, "Fitness Mean")
        _plot_fitted(ax3, fitness_max, "Fitness Max")

        plt.tight_layout()
        if save_as is not None:
            cfig = plt.gcf()
            cfig.savefig(f"plots/{save_as}.png", transparent=True)
        plt.show()

    @classmethod
    def plot_metrics_config(cls, metric: dict, ylabel: str, stds:dict = None, metric2: dict = None,  ymax:float = 1.0, save_as:str=None):
        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        fig.supxlabel("Generations")
        fig.supylabel(ylabel)
        for i, key in enumerate(metric.keys()):
            tmp_df = pd.DataFrame(metric[key]).T.describe().T

            std = tmp_df["std"] if stds is None else [sum(gen)/len(gen) for gen in stds[key]]
            if ylabel == "Fitness":

                to_plot = tmp_df["mean"]

            else:

                to_plot = tmp_df[["mean", "min"]]
            axs[i // 3, i % 3].plot(tmp_df.index, to_plot)

            axs[i // 3, i % 3].hlines(tmp_df["mean"].tail(1), 0, 400, colors="r", linestyles="dashed")


            if metric2:
                axs[i // 3, i % 3].text(y=tmp_df["mean"].tail(1),x=20, s=f"Mean: {tmp_df['mean'].tail(1).values[0]:.3f}")

                m2_df = pd.DataFrame(metric2[key]).T.describe().T
                axs[i // 3, i % 3].text(y=m2_df["mean"].tail(1), x=20, s=f"Max: {m2_df['mean'].tail(1).values[0]:.3f}")
                axs[i // 3, i % 3].hlines(m2_df["mean"].tail(1), 0, 400, colors="r",
                                          linestyles="dashed")
                axs[i // 3, i % 3].plot(m2_df.index, m2_df["mean"], label="Max")
                axs[i // 3, i % 3].fill_between(m2_df.index, m2_df["mean"] - m2_df["std"], m2_df["mean"] + m2_df["std"], alpha=0.5,
                                                color="grey")

            else:
                axs[i // 3, i % 3].text(y=tmp_df["mean"].tail(1),x=20, s=f"{tmp_df['mean'].tail(1).values[0]:.3f}")
            axs[i // 3, i % 3].fill_between(tmp_df.index, tmp_df["mean"] - std, tmp_df["mean"] + std, alpha=0.5,
                                            color="grey")


            axs[i // 3, i % 3].set_ylim([-0.05, ymax])
            axs[i // 3, i % 3].set_xlim([-5, 405])

            key = key if "None" not in key else "Prod"
            axs[i // 3, i % 3].set_title(f"Novelty Config: {key}")



        plt.tight_layout()
        if save_as is not None:
            cfig = plt.gcf()
            cfig.savefig(f"plots/{save_as}.png", transparent=True)
        plt.show()

    @classmethod
    def plot_finess_over_novelty(cls, fitness:dict, novelty:dict, save_as:str=None):
        fig, axs = plt.subplots(2, 3, figsize=(15, 7))
        fig.supxlabel("Generations")
        fig.supylabel("Fitness * Novelty")
        for i, key in enumerate(fitness.keys()):
            tmp_f_df = pd.DataFrame(fitness[key]).T.describe().T
            tmp_n_df = pd.DataFrame(novelty[key]).T.describe().T

            to_plot = tmp_f_df["mean"] * tmp_n_df["mean"]
            av = to_plot.mean()

            axs[i // 3, i % 3].plot(tmp_f_df.index, to_plot)
            axs[i // 3, i % 3].set_title(f"Novelty Config: {key}")


            print("\n"+key)
            print(f"Average: {av:.4f}")
        if save_as is not None:
            cfig = plt.gcf()
            cfig.savefig(f"plots/{save_as}.png", transparent=True)

    @classmethod
    def _get_img(cls, body, size):
        body_arr, core_pos = body.to_grid()
        body_arr = np.asarray(body_arr)
        x, y, z = body_arr.shape
        cx, cy, cz = core_pos

        img_xy, img_xz, img_yz = np.zeros((size, size)), np.zeros((size, size)), np.zeros((size, size))

        m = size // 2
        img_xy[m, m] += 1
        img_xz[m, m] += 1
        img_yz[m, m] += 1
        for xe in range(x):
            for ye in range(y):
                for ze in range(z):
                    elem = body_arr[xe][ye][ze]
                    if isinstance(elem, Brick) or isinstance(elem, ActiveHinge):
                        img_xy[xe - cx + m, ye - cy + m] += 1
                        img_xz[xe - cx + m, ze - cz + m] += 1
                        img_yz[ye - cy + m, ze - cz + m] += 1

        return img_xy.clip(max=1), img_xz.clip(max=1), img_yz.clip(max=1)

    @classmethod
    def stat_testing(cls, mm: pd.DataFrame, test:str, name_add:str= "", num_simulations:int=30):
        for idx, row in mm.iterrows():
            ltx_str = r"\begin{table}[H]\centering\begin{tabular}{l|llllll}\textbf{} & \textbf{0.0} & \textbf{0.25} & \textbf{0.5} & \textbf{0.75} & \textbf{1.0} & \textbf{None} \\ \hline{ \textbf{0.0}} & \cellcolor[HTML]{9B9B9B}i11 & i12 & i13 & i14 & i15 & i16 \\{ \textbf{0.25}} &  & \cellcolor[HTML]{9B9B9B}i22 & i23 & i24 & i25 & i26 \\{\textbf{0.5}} & {} & { } & \cellcolor[HTML]{9B9B9B}{i33} & i34 & i35 & i36 \\{\textbf{0.75}} & { } & {} & {} & \cellcolor[HTML]{9B9B9B}i44 & i45 & i46 \\{\textbf{1.0}} & { } & {} & {} &  & \cellcolor[HTML]{9B9B9B}i55 & i56 \\{\textbf{None}} & { } & { } & {\color[HTML]{9B9B9B} } &  &  & \cellcolor[HTML]{9B9B9B}{ i66}\end{tabular}\end{table}"
            lr = len(row)
            tests ={"wilcoxon": wilcoxon,
                    "ranksums": ranksums,
                    "kstest": kstest,
                    "ttest": ttest_ind
            }


            for i in range(lr):
                """
                sz = len(row[i]) // num_simulations
                corrs = []
                for a in range(num_simulations-1):
                    for b in range(a + 1, num_simulations):
                        corrs.append(tests[test](row[i][a * sz:a * sz + sz], row[i][b * sz:b * sz + sz]).pvalue)
                val = sum(corrs) / len(corrs)
                ltx_str = ltx_str.replace(f"i{i + 1}{i + 1}", f"{val:.4E}")
                """
                ltx_str = ltx_str.replace(f"i{i + 1}{i + 1}", "")
                if i == lr-1:
                    break

                for j in range(i+1, lr):
                    sz = min(len(row[i]), len(row[j]))
                    val = tests[test](row[i][:sz], row[j][:sz]).pvalue
                    ltx_str = ltx_str.replace(f"i{i+1}{j+1}", f"{val:.4E}") if val < 0.05 else ltx_str.replace(f"i{i+1}{j+1}", f"\cellcolor[HTML]{{FD6864}}{val:.4E}")

            with open(f"plots/{name_add}{idx}.txt", "w") as f:
                f.write(ltx_str)


    @classmethod
    def plot_morphological_descriptors_development(cls, databases: List[str], seperator:str, db_id: DbId = DbId("optmodular"), generations:int = 400, save_as:str="default"):
        if os.path.exists(f"plots/{save_as}_{seperator}_morph.png"):
            return None

        print(f"{seperator} - Warning: This function takes extremly long!! It is not frozen")
        cov, sym, lim = [[] for _ in range(generations)], [[] for _ in range(generations)], [[] for _ in range(generations)]
        for database in databases:
            df = load_db_novelty(database, db_id)
            vals = (df[["generation_index", "serialized_multineat_genome"]].groupby(by="generation_index")["serialized_multineat_genome"].apply(list))
            for i in range(generations):
                for genotype in vals.iloc[i]:
                    body = develop_v1(pf.deserialize(genotype))
                    try:
                        mm = MorphologicalMeasures(body)
                        cov[i].append(mm.coverage)
                        sym[i].append(mm.symmetry)
                        lim[i].append(mm.limbs)
                    except:
                        continue

        fig, axs = plt.subplots(1, 3, figsize=(15, 3))
        fig.supxlabel("Generations")

        for ax,label,data in zip(axs,["Coverage", "Symmetry", "Limbs"], [cov, sym, lim]):
            ax.set_ylabel(label)
            cmean, cmin, cmax, cstd = cls._describe_lists(data)

            avg_std = sum(cstd)/len(cstd)
            ax.plot(cmean, color="blue")
            ax.plot([], label=f"std: {avg_std:.4f}")
            #ax.plot(cmin, label="Min", color="orange")
            #ax.plot(cmax, label="Max", color="green")
            ax.fill_between(list(range(generations)),
                             [c - s if c-s >= 0 else 0 for c, s in zip(cmean, cstd)],
                             [c + s for c, s in zip(cmean, cstd)],
                             alpha=0.5, color="grey")
            ax.legend()
            ax.set_ylim([-0.05, 1.1])

        plt.tight_layout()
        plt.savefig(f"plots/{save_as}_{seperator}_morph.png", transparent=True)


    @classmethod
    def _describe_lists(cls, input: List[List]) -> Tuple[List, List, List, List]:
        ll = len(input)
        mean, mmin, mmax, std = [0]*ll, [0]*ll, [0]*ll, [0]*ll
        for i in range(ll):
            curr = input[i]
            mean[i] = sum(curr)/len(curr)
            mmin[i] = min(curr)
            mmax[i] = max(curr)
            std[i] = statistics.stdev(curr)

        return mean, mmin, mmax, std

    @classmethod
    def plot_metric_from_db(cls, database: str, metric:str = "Novelty", db_id: DbId = DbId("optmodular"), *_) -> float:
        """
        Plot fitness as described at the top of this file.
        :param database: Database where the data is stored.
        :param db_id: Id of the evolutionary process to plot.
        """

        df = load_db_novelty(database, db_id) if metric == "Novelty" else load_db_fitness(database, db_id)

        nvlt = (
            df[["generation_index", "value"]]
            .groupby(by="generation_index")
            .describe()["value"])

        # bxpl_data = [n["value"].values for _, n in nvlt]

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle(f"{database}")

        ax.set_xlabel("Generations")
        ax.set_ylabel(metric)
        ax.plot(nvlt[["max", "mean", "min"]], label=["Max", "Mean", "Min"])
        if metric == "Novelty":
            ax.set_ylim([-0.05, 1.05])
        # ax.violinplot(bxpl_data, positions=list(range(1,len(bxpl_data)+1)))
        ax.legend()
        plt.show()
        return nvlt["mean"].mean()

