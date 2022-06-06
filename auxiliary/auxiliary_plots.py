import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import SubplotSpec
import seaborn as sns


def figure_1(age_per, school_dist_per):

    df_1 = age_per.copy()
    df_2 = school_dist_per.copy()
    df_1.loc[:, "inschool_india":"inschool_bihar_female"] = (
        age_per.loc[:, "inschool_india":"inschool_bihar_female"] * 100
    )
    df_2.loc[:, "highschool_india":"highschool_bihar_female"] = (
        school_dist_per.loc[:, "highschool_india":"highschool_bihar_female"] * 100
    )

    def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f"\n{title}\n\n", loc="left")
        # hide subplot
        row.set_frame_on(False)
        row.axis("off")

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=80)

    axs[0, 0].plot(
        df_1["age"],
        df_1["inschool_india_male"],
        label="Male",
        color="blue",
        linestyle="dashed",
    )
    axs[0, 0].plot(df_1["age"], df_1["inschool_india"], label="All", color="black")
    axs[0, 0].plot(
        df_1["age"],
        df_1["inschool_india_female"],
        label="Female",
        color="red",
        linestyle="dotted",
    )
    axs[0, 0].set_title("India")
    axs[0, 0].set_ylabel("Percent")
    axs[0, 0].set_xlabel("Age (years)")
    axs[0, 0].set_ylim([40, 100])
    axs[0, 0].legend(loc=3)
    fig.suptitle("FIGURE 1", fontsize=12)

    axs[0, 1].plot(
        df_1["age"], df_1["inschool_bihar_male"], color="blue", linestyle="dashed"
    )
    axs[0, 1].plot(df_1["age"], df_1["inschool_bihar"], color="black")
    axs[0, 1].plot(
        df_1["age"], df_1["inschool_bihar_female"], color="red", linestyle="dotted"
    )
    axs[0, 1].set_title("Bihar")
    axs[0, 1].set_ylabel("Percent")
    axs[0, 1].set_xlabel("Age (years)")
    axs[0, 1].set_ylim([40, 100])

    axs[1, 0].plot(
        df_2["secondarydist"],
        df_2["highschool_india_male"],
        label="Male",
        color="blue",
        linestyle="dashed",
    )
    axs[1, 0].plot(
        df_2["secondarydist"], df_2["highschool_india"], label="All", color="black"
    )
    axs[1, 0].plot(
        df_2["secondarydist"],
        df_2["highschool_india_female"],
        label="Female",
        color="red",
        linestyle="dotted",
    )
    axs[1, 0].set_title("India")
    axs[1, 0].set_ylabel("Percent")
    axs[1, 0].set_xlabel("Distance to secondary school (km)")
    axs[1, 0].set_ylim([10, 70])
    axs[1, 0].set_xticks(np.arange(0, 20, 5))
    axs[1, 0].legend(loc=3)

    axs[1, 1].plot(
        df_2["secondarydist"],
        df_2["highschool_bihar_male"],
        color="blue",
        linestyle="dashed",
    )
    axs[1, 1].plot(df_2["secondarydist"], df_2["highschool_bihar"], color="black")
    axs[1, 1].plot(
        df_2["secondarydist"],
        df_2["highschool_bihar_female"],
        color="red",
        linestyle="dotted",
    )
    axs[1, 1].set_title("Bihar")
    axs[1, 1].set_xlabel("Distance to secondary school (km)")
    axs[1, 1].set_ylabel("Percent")
    axs[1, 1].set_ylim([10, 70])
    axs[1, 1].set_xticks(np.arange(0, 20, 5))

    grid = plt.GridSpec(2, 2)
    create_subtitle(fig, grid[0, ::], "Panel A. Enrollment in school by age and gender")
    create_subtitle(
        fig,
        grid[1, ::],
        "Panel B. 16- and 17-year-olds enrolled in or ecompleted grade 9 by distance and gender",
    )
    fig.tight_layout()
    fig.set_facecolor("w")

    return fig


def figure_2(dist_data_per):
    df = dist_data_per.copy()
    df = df.drop(columns=["_merge"])

    fig, axs = plt.subplots(3, figsize=(14, 18), dpi=70)

    axs[0].plot(df["longdistgroup"], df["dd10"], color="black")
    axs[0].set_ylim([-0.1, 0.3])
    axs[0].set_xticks(np.arange(0, 30, 5))
    axs[0].set_title(
        "\nPanel A. Bihar double difference by distance to secondary school", loc="left"
    )
    axs[0].set_ylabel(
        "Double difference\n change in girls' enrollment-\n change in boys' enrollment"
    )
    axs[0].set_xlabel("Distance to secondary school (km)\n")
    axs[0].hlines(y=0, xmin=0, xmax=25, linewidth=1, color="black")
    # Define the confidence interval
    ci_1 = 0.1 * np.std(df["dd10"]) / np.mean(df["dd10"])
    axs[0].fill_between(
        df["longdistgroup"],
        (df["dd10"] - ci_1),
        (df["dd10"] + ci_1),
        color="blue",
        alpha=0.1,
    )
    fig.suptitle(
        "FIGURE 2. Non Parametric Double and Triple Difference Estimates of Impact\nof the Cycle Program (by distance to nearest secondary school)",
        fontsize=12,
    )

    axs[1].plot(df["longdistgroup"], df["dd20"], color="black")
    axs[1].set_ylim([-0.1, 0.2])
    axs[1].set_xticks(np.arange(0, 30, 5))
    axs[1].set_title(
        "\nPanel B. Jharkhand double difference by distance to secondary school",
        loc="left",
    )
    axs[1].set_ylabel(
        "Double difference\n change in girls' enrollment-\n change in boys' enrollment"
    )
    axs[1].set_xlabel("Distance to secondary school (km)\n\n")
    axs[1].hlines(y=0, xmin=0, xmax=25, linewidth=1, color="black")
    ci_2 = 0.1 * np.std(df["dd20"]) / np.mean(df["dd20"])
    axs[1].fill_between(
        df["longdistgroup"],
        (df["dd20"] - ci_2),
        (df["dd20"] + ci_2),
        color="blue",
        alpha=0.1,
    )

    axs[2].plot(df["longdistgroup"], df["diff"], color="black")
    axs[2].set_ylim([-0.1, 0.2])
    axs[2].set_xticks(np.arange(0, 30, 5))
    axs[2].set_title(
        "\nPanel C. Triple difference by distance to secondary school", loc="left"
    )
    axs[2].set_ylabel(
        "Triple difference\n double difference in Bihar - \n double difference in Jharkhand"
    )
    axs[2].set_xlabel("Distance to secondary school (km)\n\n")
    axs[2].hlines(y=0, xmin=0, xmax=25, linewidth=1, color="black")
    ci_3 = 0.1 * np.std(df["diff"]) / np.mean(df["diff"])
    axs[2].fill_between(
        df["longdistgroup"],
        (df["diff"] - ci_3),
        (df["diff"] + ci_2),
        color="blue",
        alpha=0.1,
    )

    return fig


def figure_3(dlhs_figure_A_1):
    df_1 = dlhs_figure_A_1.loc[dlhs_figure_A_1["state"] == "Bihar"]
    df_2 = dlhs_figure_A_1.loc[dlhs_figure_A_1["state"] == "Jharkhand"]

    fig, axs = plt.subplots(1, 2, figsize=(10, 8), dpi=70)

    axs[0].hist(df_1["secondarydist"], density=True, bins=22, color="slateblue")
    axs[1].hist(df_2["secondarydist"], density=True, bins=22, color="slateblue")
    axs[0].set_ylabel("Density")
    axs[0].set_xlabel("Distance to Secondary School (km)")
    axs[1].set_ylabel("Density")
    axs[1].set_xlabel("Distance to Secondary School (km)")
    axs[0].set_title("Bihar")
    axs[1].set_title("Jharkhand")
    fig.suptitle(
        "FIGURE A.1: Distribution of Villages by Distance to Nearest Secondary School",
        fontsize=12,
    )

    return fig


def fig_4(exam_data):
    df = exam_data.loc[exam_data["gender"] == 2]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), dpi=70)
    ax1.bar(df['year'], df['appear_tot'], color ='maroon',
            width = 0.4)
    ax1.set_xlabel("years")
    ax1.set_ylabel("Total appearance in SSC exams")
    ax1.set_title("Females appeared for SSC exams")

    ax2.bar(df['year'], df['pass_tot'], color ='maroon',
            width = 0.4)
    ax2.set_xlabel("years")
    ax2.set_ylabel("Total pass in SSC exams")
    ax2.set_title("Females passed  SSC exams")
    fig.suptitle(
        "FIGURE 3",
        fontsize=12,
    )


    return fig