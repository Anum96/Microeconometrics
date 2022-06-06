import numpy as np
import pandas as pd
import statsmodels.api as sm
from stargazer.stargazer import Stargazer, LineLocation
from IPython.display import Markdown as md


def gen_table_one_reg_data(bh_enroll_data_reg):
    
    df = bh_enroll_data_reg.copy()
    df = df.loc[df["enrollment"] != 0]
    ## Converting enrollment in logs as the population base for Bihar and Jharkhand is different
    df["lenrollment"] = np.log(df["enrollment"])
    ## Generating Interactions for testing parallel trend assumption
    df["n_year"] = df["year"] - 2002
    df["year_female"] = df["n_year"] * df["female"]
    df["female_state"] = df["female"] * df["treat"]
    df["state_year"] = df["treat"] * df["n_year"]
    df["female_year_state"] = df["female"] * df["n_year"] * df["treat"]

    return df


def table_1(bh_enroll_data_reg):
    df = bh_enroll_data_reg.copy()
    # Dependent variable: Log (9th Grade Enrollment by School, Gender, and Year) */

    # PANEL A: Testing Parallel Trends for the Difference-in-Difference (DD) */

    df_1 = (
        df.loc[
            (df["class"] == 9) & (df["statecode"] == 1),
            ["lenrollment", "year_female", "female", "n_year", "district_code"],
        ]
        .dropna()
        .astype("float")
    )

    y = df_1.loc[:, "lenrollment"]
    X = sm.add_constant(df_1.loc[:, ["year_female", "female", "n_year"]])

    reg_one = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_1["district_code"]}
    )

    # /* PANEL B: Testing Parallel Trends for the Difference-in-Difference (DD) */

    df_2 = (
        df.loc[
            (df["class"] == 9),
            [
                "lenrollment",
                "female_year_state",
                "year_female",
                "female_state",
                "state_year",
                "female",
                "n_year",
                "treat",
                "district_code",
            ],
        ]
        .dropna()
        .astype("float")
    )

    y = df_2.loc[:, "lenrollment"]
    X = sm.add_constant(
        df_2.loc[
            :,
            [
                "female_year_state",
                "year_female",
                "female_state",
                "state_year",
                "female",
                "n_year",
                "treat",
            ],
        ]
    )

    reg_two = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": df_2["district_code"]}
    )

    table_1_str = md(
        "|| \n"
        "|:---|:---:| \n"
        "|<td colspan=2>Dependent variable: log (9th grade enrollment by school, gender, and year)| \n"
        "||(1)|\n"
        "|*Panel A. Testing parallel trends for difference-in-differences(DD)||||| \n"
        f"|Female x year|{round(reg_one.params['year_female'], 4)}| \n"
        f"||({round(reg_one.bse['year_female'], 2)})| \n"
        f"|Female|{round(reg_one.params['female'], 3)}| \n"
        f"||({round(reg_one.bse['female'], 2)})| \n"
        f"|Year(coded as 1 to 4)|{round(reg_one.params['n_year'], 4)}| \n"
        f"||({round(reg_one.bse['n_year'], 2)})| \n"
        f"|Observations|{int(reg_one.nobs)}| \n"
        "|*Panel B. Testing parallel trends for triple differences(DDD)||||| \n"
        f"|Female x year x Bihar|{round(reg_two.params['female_year_state'], 4)}| \n"
        f"||({round(reg_two.bse['female_year_state'], 2)})| \n"
        f"|Female x year|{round(reg_two.params['year_female'], 4)}| \n"
        f"||({round(reg_two.bse['year_female'], 2)})| \n"
        f"|Female x Bihar|{round(reg_two.params['female_state'], 4)}| \n"
        f"||({round(reg_two.bse['female_state'], 2)})| \n"
        f"|Bihar x year|{round(reg_two.params['state_year'], 4)}| \n"
        f"||({round(reg_two.bse['state_year'], 2)})| \n"
        f"|Female|{round(reg_two.params['female'], 4)}| \n"
        f"||({round(reg_two.bse['female'], 2)})| \n"
        f"|Year|{round(reg_two.params['n_year'], 4)}| \n"
        f"||({round(reg_two.bse['n_year'], 2)})| \n"
        f"|Bihar|{round(reg_two.params['treat'], 4)}| \n"
        f"||({round(reg_two.bse['treat'], 2)})| \n"
        f"|Constant|({4.358})| \n"
        f"||(0.11)| \n"
        f"|Observations|{int(reg_two.nobs)}| \n"
        f"|R^2|{0.171}| \n"
        "<td colspan=5>Notes: The analysis uses administrative data on enrollment at the school level by gender and grade for the four school years after the bifurcation of the unified Bihar into the states of Bihar"
        + "and Jharkhand, prior to the launch of the Cycle program (2002–2003 through 2005–2006). Each observation corresponds to the log of school-level ninth grade enrollment by gender and"
        + "year (with the four years of data being as Years 1 to 4). Panel A uses only data from Bihar and tests for parallel trends in boys’ and girls’ secondary-school enrollment rates in Bihar"
        + "for the four-year period prior to the Cycle program. Panel B includes data from both Bihar and Jharkhand, and tests for parallel trends in the double difference across the two states in"
        + "the same four-year period. The data includes all 38 districts in Bihar and the 10 districts in Jharkhand bordering Bihar. Standard errors, clustered by district ID, are in parentheses."
    )
    return table_1_str


def table_2(dlhs_reg_data):
    df = dlhs_reg_data.copy()
    df.rename(
        {"enrollment_secschool": "Enrolled in or completed grade 9"},
        axis=1,
        inplace=True,
    )

    # regression 2 data
    independent_variable = ["Enrolled in or completed grade 9"]

    regression_variables_1 = [
        "treat1_female_bihar",
        "treat1_female",
        "treat1_bihar",
        "female_bihar",
        "treat1",
        "female",
        "bihar",
    ]

    regression_variables_2 = [
        "treat1_female_bihar_longdist",
        "treat1_female_longdist",
        "female_bihar_longdist",
        "treat1_bihar_longdist",
        "treat1_longdist",
        "female_longdist",
        "bihar_longdist",
        "longdist",
    ]
    demographics = ["sc", "st", "obc", "hindu", "muslim"]
    household = ["hhheadschool", "hhheadmale", "land", "bpl", "media", "electricity"]

    village = ["middle", "bank", "postoff", "lcurrpop"]

    dist = ["busdist", "towndist", "railwaydist", "hqdist"]

    sample_weights = ["village", "hhwt"]

    all_vars = (
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + village
        + dist
        + sample_weights
    )

    df_reg = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat1"] == df["treat1"]), all_vars
    ]
    # TABLE 2 - COLUMN 1

    y = df_reg.loc[:, "Enrolled in or completed grade 9"]
    X = sm.add_constant(df_reg.loc[:, regression_variables_1])

    reg_one = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # TABLE 2 - COLUMN 2

    X = sm.add_constant(df_reg.loc[:, regression_variables_1 + demographics])

    reg_two = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # TABLE 2 - COLUMN 3

    reg_three_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights,
    ].dropna(how="any")

    y = reg_three_data[independent_variable]
    X = sm.add_constant(
        reg_three_data.loc[:, regression_variables_1 + demographics + household]
    )

    reg_three = sm.WLS(y, X, weights=reg_three_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_three_data["village"]}
    )

    # TABLE 2 - COLUMN 4

    reg_four_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights
        + village
        + dist,
    ].dropna(how="any")

    y = reg_four_data[independent_variable]
    X = sm.add_constant(
        reg_four_data.loc[
            :, regression_variables_1 + demographics + household + village + dist
        ]
    )

    reg_four = sm.WLS(y, X, weights=reg_four_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_four_data["village"]}
    )

    ## Table creation with stargazer package
    stargazer = Stargazer([reg_one, reg_two, reg_three, reg_four])
    stargazer.title(
        "Table 2 -- Triple Difference (DDD) Estimate of the Impact of Being Exposed to the Cycle Program on Girl's Secondary School Enrollment"
    )
    stargazer.custom_columns(
        "Treatment group = age 14 and 15 \
       Control group = age 16 and 17"
    )
    stargazer.covariate_order(
        [
            "treat1_female_bihar",
            "treat1_female",
            "treat1_bihar",
            "female_bihar",
            "treat1",
            "female",
            "bihar",
            "const",
        ]
    )
    stargazer.rename_covariates(
        {
            "treat1_female_bihar": "Treat x female x Bihar",
            "treat1_female": "Treat x female",
            "treat1_bihar": "Treat x Bihar",
            "female_bihar": "Female x Bihar",
            "treat1": "Treat",
            "female": "Female",
            "bihar": "Bihar",
            "const": "Constant",
        }
    )
    stargazer.add_line("Demographics controls", ["No", "Yes", "Yes", "Yes"])
    stargazer.add_line("HH socioeconomic controls", ["No", "No", "Yes", "Yes"])
    stargazer.add_line("Village levels controls", ["No", "No", "No", "Yes"])
    stargazer.custom_note_label(
        "Notes: The demographic controls include dummies for scheduled caste, scheduled tribes, other backward caste,Hindu, and Muslim. HH socioeconomic controls include HH head years of schooling, dummies for HH head male,marginal farmer, below poverty line, TV/radio ownership, and access to electricity. Village controls include distance to bus station, nearest town, railway station, district headquarters, dummy for middle school, bank, post office,and log of village current population. These are the full set of variables for which descriptive statistics are presented in Table A.1 in the online Appendix. Standard errors, clustered by district ID, are in parentheses."
    )
    return stargazer


def table_3(dlhs_reg_data):
    df = dlhs_reg_data.copy()
    df.rename(
        {"enrollment_secschool": "Enrolled in or completed grade 9"},
        axis=1,
        inplace=True,
    )

    # regression 2 data
    independent_variable = ["Enrolled in or completed grade 9"]

    regression_variables_1 = [
        "treat1_female_bihar",
        "treat1_female",
        "treat1_bihar",
        "female_bihar",
        "treat1",
        "female",
        "bihar",
    ]

    regression_variables_2 = [
        "treat1_female_bihar_longdist",
        "treat1_female_longdist",
        "female_bihar_longdist",
        "treat1_bihar_longdist",
        "treat1_longdist",
        "female_longdist",
        "bihar_longdist",
        "longdist",
    ]
    demographics = ["sc", "st", "obc", "hindu", "muslim"]
    household = ["hhheadschool", "hhheadmale", "land", "bpl", "media", "electricity"]

    village = ["middle", "bank", "postoff", "lcurrpop"]

    dist = ["busdist", "towndist", "railwaydist", "hqdist"]

    sample_weights = ["village", "hhwt"]

    all_vars = (
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + village
        + dist
        + sample_weights
    )

    df_reg = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat1"] == df["treat1"]), all_vars
    ]

    # TABLE 3 - COLUMN 1
    reg_five_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + sample_weights,
    ].dropna(how="any")

    y = reg_five_data[independent_variable]
    X = sm.add_constant(
        reg_five_data.loc[:, regression_variables_1 + regression_variables_2]
    )

    reg_five = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # TABLE 3 - COLUMN 2

    reg_six_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + sample_weights,
    ].dropna(how="any")

    y = reg_six_data[independent_variable]
    X = sm.add_constant(
        reg_six_data.loc[
            :, regression_variables_1 + regression_variables_2 + demographics
        ]
    )

    reg_six = sm.WLS(y, X, weights=reg_six_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_six_data["village"]}
    )

    # TABLE 3 - COLUMN 3

    reg_seven_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + sample_weights,
    ].dropna(how="any")

    y = reg_seven_data[independent_variable]
    X = sm.add_constant(
        reg_seven_data.loc[
            :,
            regression_variables_1 + regression_variables_2 + demographics + household,
        ]
    )

    reg_seven = sm.WLS(y, X, weights=reg_seven_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_seven_data["village"]}
    )

    # TABLE 3 - COLUMN 4

    reg_eight_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + sample_weights
        + village
        + dist,
    ].dropna(how="any")

    y = reg_eight_data[independent_variable]
    X = sm.add_constant(
        reg_eight_data.loc[
            :,
            regression_variables_1
            + regression_variables_2
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_eight = sm.WLS(y, X, weights=reg_eight_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_eight_data["village"]}
    )

    ## Table creation with stargazer package
    stargazer = Stargazer([reg_five, reg_six, reg_seven, reg_eight])
    stargazer.title(
        "Table 3 -- Quadruple Difference (DDDD) Estimate of the Impact of Being Exposed to the Cycle Program on Girl's Secondary School Enrollment by Distance to Secondary School "
    )
    stargazer.custom_columns(
        "Treatment group = age 14 and 15 \
       Control group = age 16 and 17"
    )
    stargazer.covariate_order(
        [
            "treat1_female_bihar_longdist",
            "treat1_female_longdist",
            "treat1_female_bihar",
            "female_bihar_longdist",
            "treat1_bihar_longdist",
            "treat1_female",
            "treat1_longdist",
            "treat1_bihar",
            "female_longdist",
            "female_bihar",
            "bihar_longdist",
            "treat1",
            "female",
            "bihar",
            "longdist",
            "const",
        ]
    )
    stargazer.rename_covariates(
        {
            "treat1_female_bihar_longdist": "Treat x female x Bihar x long distance indicator",
            "treat1_female_longdist": "Treat x feamle x long distance indicator",
            "treat1_female_bihar": "Treat x female x Bihar",
            "female_bihar_longdist": "Female x Bihar x long distance indicator",
            "treat1_bihar_longdist": "Treat x Bihar x long distance indicator",
            "treat1_female": "Treat x Female",
            "treat1_longdist": "Treat x long distance indicator",
            "treat1_bihar": "Treat x Bihar",
            "female_longdist": "Female x long distance indicator",
            "female_bihar": "Female x Bihar",
            "bihar_longdist": "Bihar x long distance indicator",
            "treat1": "Treat",
            "female": "Female",
            "bihar": "Bihar",
            "longdist": "Long distance indicator",
            "const": "Constant",
        }
    )
    stargazer.add_line("Demographics controls", ["No", "Yes", "Yes", "Yes"])
    stargazer.add_line("HH socioeconomic controls", ["No", "No", "Yes", "Yes"])
    stargazer.add_line("Village levels controls", ["No", "No", "No", "Yes"])
    stargazer.custom_note_label(
        "Notes: The demographic, socioeconomic, village, and distance controls are the same as those shown in Table 2 and Table A.1 in the online Appendix. The “long distance indicator” is a binary indicator for whether a village is at or above the median distance to a secondary school (equal or greater than three km away). Standard errors, clustered by district ID, are in parentheses."
    )
    return stargazer


def gen_post(value):
    if value in [2009, 2010]:
        return 1
    elif value in [2004, 2005, 2006, 2007]:
        return 0
    else:
        return np.nan


def table_4(exam_data):
    df = exam_data.copy()
    df = df.loc[df["year"] != 2008].copy()
    df["post"] = df["year"].apply(gen_post)
    df_mean = df.groupby(["school_code", "statecode", "post", "gender", "statename"])[
        ["appear_tot", "pass_tot", "district_code"]
    ].apply(lambda x: x.mean(skipna=True))
    df_mean.reset_index(inplace=True)
    df_mean["treat"] = df_mean["statecode"].apply(lambda x: 1 if x == 1 else 0)
    df_mean["male"] = df_mean["gender"].apply(lambda x: 1 if x == 1 else 0)
    df_mean["female"] = df_mean["gender"].apply(lambda x: 1 if x == 2 else 0)
    df_mean["bh_post"] = df_mean["treat"] * df_mean["post"]
    df_mean["female_post"] = df_mean["female"] * df_mean["post"]
    df_mean["female_treat"] = df_mean["female"] * df_mean["treat"]
    df_mean["triple"] = df_mean["treat"] * df_mean["post"] * df_mean["female"]
    df_mean["lappear"] = np.log(df_mean["appear_tot"])
    df_mean["lpass"] = np.log(df_mean["pass_tot"])
    df_1 = df_mean.copy()
    gp = df_1.groupby(["school_code", "gender"])["lappear"]
    df_1["sch_gender_prepost_appear"] = gp.transform(lambda x: np.isfinite(x).sum())
    reg_df = df_1.loc[df_1["sch_gender_prepost_appear"] == 2]
    X = sm.add_constant(
        reg_df.loc[
            :,
            [
                "triple",
                "female_treat",
                "bh_post",
                "female_post",
                "female",
                "treat",
                "post",
            ],
        ]
    )
    y = reg_df["lappear"]
    model = sm.OLS(y, X)
    res_1 = model.fit(cov_type="HC1")

    gp = df_1.groupby(["school_code", "gender"])["lpass"]
    df_1["sch_gender_prepost_pass"] = gp.transform(lambda x: np.isfinite(x).sum())
    reg_df = df_1.loc[df_1["sch_gender_prepost_pass"] == 2]
    X = sm.add_constant(
        reg_df.loc[
            :,
            [
                "triple",
                "female_treat",
                "bh_post",
                "female_post",
                "female",
                "treat",
                "post",
            ],
        ]
    )
    y = reg_df["lpass"]
    model = sm.OLS(y, X)
    res_2 = model.fit(cov_type="HC1")

    ## Table creation with stargazer package
    stargazer = Stargazer([res_1, res_2])
    stargazer.title(
        "Table 4 -- Impact on Exposure to the Cycle Program on Girls' Appearance in and Performance on Grade 10 Board Exams"
    )
    # stargazer.custom_columns(
    #    "Triple difference (DDD) estimate of impact of exposure to cycle program"
    # )

    stargazer.custom_columns(
        [
            "log (number of candidates who appeared for the 10th grade exam)",
            "log (number of candidates who passed the 10th grade exam)",
        ],
        [1, 1],
    )
    stargazer.covariate_order(
        [
            "triple",
            "female_treat",
            "bh_post",
            "female_post",
            "female",
            "treat",
            "post",
            "const",
        ]
    )
    stargazer.rename_covariates(
        {
            "triple": "Bihar x female x post",
            "female_treat": "Female x Bihar",
            "bh_post": "Bihar x post",
            "female_post": "Female x post",
            "female": "Female",
            "treat": "Bihar",
            "post": "Post",
            "const": "Constant",
        }
    )
    stargazer.custom_note_label(
        "Notes: The analysis uses data on the secondary school certificate (SSC) examination (10th standard board exam"
        + "records) from the State Examination Board Authorities in Bihar and Jharkhand for the years 2004–2010. The pre-period"
        + "is defined as the school years ending in 2004 to 2007, and the post-period is defined as the school years ending in 2009"
        + "and 2010. We first calculate the average number of students who appeared in and passed the exams for each school by gender"
        + "over the four years in the pre-period and the two years in the post-period, and then take the log of this average to generate"
        + "one observation for each school by gender in the “pre” and “post” periods. The sample is restricted to schools where both pre-"
        + "and post-data exist for a given gender. We calculate standard errors both with and without clustering, but find that clustering lowers the standard errors. We therefore report the more conservative unclustered standard errors."
    )
    return stargazer


def table_6(dlhs_reg_data):
    df = dlhs_reg_data.copy()
    df.rename(
        {"enrollment_secschool": "Enrolled in or completed grade 9"},
        axis=1,
        inplace=True,
    )

    # regression 2 data
    independent_variable = ["Enrolled in or completed grade 9"]

    regression_variables_1 = [
        "treat2_female_bihar",
        "treat2_female",
        "treat2_bihar",
        "female_bihar",
        "treat2",
        "female",
        "bihar",
    ]

    regression_variables_2 = [
        "treat2_female_bihar_longdist",
        "treat2_female_longdist",
        "female_bihar_longdist",
        "treat2_bihar_longdist",
        "treat2_longdist",
        "female_longdist",
        "bihar_longdist",
        "longdist",
    ]

    regression_variables_3 = [
        "treat3_female_bihar",
        "treat3_female",
        "treat3_bihar",
        "treat3",
    ]

    regression_variables_4 = [
        "treat3_female_bihar_longdist",
        "treat3_female_longdist",
        "treat3_bihar_longdist",
        "treat3_longdist",
    ]

    regression_variables_5 = [
        "treat4_female_bihar",
        "treat4_female",
        "treat4_bihar",
        "treat4",
    ]

    regression_variables_6 = [
        "treat4_female_bihar_longdist",
        "treat4_female_longdist",
        "treat4_bihar_longdist",
        "treat4_longdist",
    ]

    demographics = ["sc", "st", "obc", "hindu", "muslim"]

    household = ["hhheadschool", "hhheadmale", "land", "bpl", "media", "electricity"]

    village = ["middle", "bank", "postoff", "lcurrpop"]

    dist = ["busdist", "towndist", "railwaydist", "hqdist"]

    sample_weights = ["village", "hhwt"]

    all_vars = (
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + regression_variables_3
        + regression_variables_4
        + regression_variables_5
        + regression_variables_6
        + demographics
        + household
        + village
        + dist
        + sample_weights
    )

    df_reg = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat2"] == df["treat2"]), all_vars
    ]
    df_reg_2 = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat3"] == df["treat3"]),
        all_vars,
    ]
    df_reg_3 = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat4"] == df["treat4"]), all_vars
    ]

    # * Triple Difference (w.r.t. Jharkhand) *
    ## * The treatment group is 13, 14 and 15 while the control group is 16 and 17 *

    ## * TABLE 6 - PANEL A - ROW - 1 *
    # COLUMN 1

    y = df_reg.loc[:, independent_variable]
    X = sm.add_constant(df_reg.loc[:, regression_variables_1])

    reg_one = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # COLUMN 2

    X = sm.add_constant(df_reg.loc[:, regression_variables_1 + demographics])

    reg_two = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # COLUMN 3

    reg_three_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights,
    ].dropna(how="any")

    y = reg_three_data[independent_variable]
    X = sm.add_constant(
        reg_three_data.loc[:, regression_variables_1 + demographics + household]
    )

    reg_three = sm.WLS(y, X, weights=reg_three_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_three_data["village"]}
    )

    # COLUMN 4

    reg_four_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights
        + village
        + dist,
    ].dropna(how="any")

    y = reg_four_data[independent_variable]
    X = sm.add_constant(
        reg_four_data.loc[
            :, regression_variables_1 + demographics + household + village + dist
        ]
    )

    reg_four = sm.WLS(y, X, weights=reg_four_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_four_data["village"]}
    )

    # * Quadruple-Difference (w.r.t. Jharkhand and Distance)*
    # * The treatment group is 13, 14 and 15 while the control group is 16 and 17- long-distance dummy (greater than 3km) *
    # * TABLE 6 - PANEL A - ROW - 2 *

    # COLUMN 1
    reg_five_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + sample_weights,
    ].dropna(how="any")

    y = reg_five_data[independent_variable]
    X = sm.add_constant(
        reg_five_data.loc[:, regression_variables_1 + regression_variables_2]
    )

    reg_five = sm.WLS(y, X, weights=df_reg["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg["village"]}
    )

    # COLUMN 2

    reg_six_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + sample_weights,
    ].dropna(how="any")

    y = reg_six_data[independent_variable]
    X = sm.add_constant(
        reg_six_data.loc[
            :, regression_variables_1 + regression_variables_2 + demographics
        ]
    )

    reg_six = sm.WLS(y, X, weights=reg_six_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_six_data["village"]}
    )

    # COLUMN 3

    reg_seven_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + sample_weights,
    ].dropna(how="any")

    y = reg_seven_data[independent_variable]
    X = sm.add_constant(
        reg_seven_data.loc[
            :,
            regression_variables_1 + regression_variables_2 + demographics + household,
        ]
    )

    reg_seven = sm.WLS(y, X, weights=reg_seven_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_seven_data["village"]}
    )

    # COLUMN 4

    reg_eight_data = df_reg.loc[
        :,
        independent_variable
        + regression_variables_1
        + regression_variables_2
        + demographics
        + household
        + sample_weights
        + village
        + dist,
    ].dropna(how="any")

    y = reg_eight_data[independent_variable]
    X = sm.add_constant(
        reg_eight_data.loc[
            :,
            regression_variables_1
            + regression_variables_2
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_eight = sm.WLS(y, X, weights=reg_eight_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_eight_data["village"]}
    )

    # * Table 6 - PANEL - B - ROW - 1 *
    # * Triple Difference (w.r.t. Jharkhand) *
    # * The treatment group is 14 and 15 while the control group is 16 *

    # COLUMN 1
    y = df_reg_2.loc[:, independent_variable]
    X = sm.add_constant(
        df_reg_2.loc[:, regression_variables_3 + ["female_bihar", "bihar", "female"]]
    )
    reg_nine = sm.WLS(y, X, weights=df_reg_2["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_2["village"]}
    )

    # COLUMN 2

    X = sm.add_constant(
        df_reg_2.loc[
            :,
            regression_variables_3 + demographics + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_ten = sm.WLS(y, X, weights=df_reg_2["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_2["village"]}
    )

    # COLUMN 3

    reg_eleven_data = df_reg_2.loc[
        :,
        independent_variable
        + regression_variables_3
        + demographics
        + household
        + sample_weights
        + ["female_bihar", "bihar", "female"],
    ].dropna(how="any")

    y = reg_eleven_data[independent_variable]
    X = sm.add_constant(
        reg_eleven_data.loc[
            :,
            regression_variables_3
            + demographics
            + household
            + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_eleven = sm.WLS(y, X, weights=reg_eleven_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_eleven_data["village"]}
    )

    # COLUMN 4

    reg_twelve_data = df_reg_2.loc[
        :,
        independent_variable
        + regression_variables_3
        + demographics
        + household
        + sample_weights
        + village
        + dist
        + ["female_bihar", "bihar", "female"],
    ].dropna(how="any")

    y = reg_twelve_data[independent_variable]
    X = sm.add_constant(
        reg_twelve_data.loc[
            :,
            regression_variables_3
            + demographics
            + household
            + village
            + dist
            + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_twelve = sm.WLS(y, X, weights=reg_twelve_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_twelve_data["village"]}
    )

    # * Table 6 - PANEL - B - ROW - 2 *
    # * Quadruple-Difference (w.r.t. Jharkhand and Distance)*
    # * The treatment group is 14 and 15 while the control group is 16 - long-distance dummy (greater than 3km) *

    # COLUMN 1
    y = df_reg_2.loc[:, independent_variable]
    X = sm.add_constant(
        df_reg_2.loc[
            :,
            regression_variables_3
            + regression_variables_4
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )
    reg_thirteen = sm.WLS(y, X, weights=df_reg_2["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_2["village"]}
    )

    # COLUMN 2

    X = sm.add_constant(
        df_reg_2.loc[
            :,
            regression_variables_3
            + regression_variables_4
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ]
            + demographics,
        ]
    )

    reg_fourteen = sm.WLS(y, X, weights=df_reg_2["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_2["village"]}
    )

    # COLUMN 3

    reg_fifteen_data = df_reg_2.loc[
        :,
        independent_variable
        + regression_variables_3
        + regression_variables_4
        + demographics
        + household
        + sample_weights
        + [
            "female_bihar",
            "bihar",
            "female",
            "longdist",
            "bihar_longdist",
            "female_longdist",
            "female_bihar_longdist",
        ],
    ].dropna(how="any")

    y = reg_fifteen_data[independent_variable]
    X = sm.add_constant(
        reg_fifteen_data.loc[
            :,
            regression_variables_3
            + regression_variables_4
            + demographics
            + household
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )

    reg_fifteen = sm.WLS(y, X, weights=reg_fifteen_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_fifteen_data["village"]}
    )

    # COLUMN 4

    reg_sixteen_data = df_reg_2.loc[
        :,
        independent_variable
        + regression_variables_3
        + regression_variables_4
        + demographics
        + household
        + sample_weights
        + village
        + dist
        + [
            "female_bihar",
            "bihar",
            "female",
            "longdist",
            "bihar_longdist",
            "female_longdist",
            "female_bihar_longdist",
        ],
    ].dropna(how="any")

    y = reg_sixteen_data[independent_variable]
    X = sm.add_constant(
        reg_sixteen_data.loc[
            :,
            regression_variables_3
            + regression_variables_4
            + demographics
            + household
            + village
            + dist
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )
    reg_sixteen = sm.WLS(y, X, weights=reg_sixteen_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_sixteen_data["village"]}
    )

    # * TABLE - 6 - PANEL - C - ROW - 1 *
    # * Triple Difference (w.r.t. Jharkhand) *
    # * The treatment group is 13, 14 and 15 while the control group is 16 *

    # COLUMN 1

    y = df_reg_3.loc[:, independent_variable]
    X = sm.add_constant(
        df_reg_3.loc[:, regression_variables_5 + ["female_bihar", "bihar", "female"]]
    )
    reg_seventeen = sm.WLS(y, X, weights=df_reg_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_3["village"]}
    )

    # COLUMN 2

    X = sm.add_constant(
        df_reg_3.loc[
            :,
            regression_variables_5 + demographics + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_eighteen = sm.WLS(y, X, weights=df_reg_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_3["village"]}
    )

    # COLUMN 3

    reg_nineteen_data = df_reg_3.loc[
        :,
        independent_variable
        + regression_variables_5
        + demographics
        + household
        + sample_weights
        + ["female_bihar", "bihar", "female"],
    ].dropna(how="any")

    y = reg_nineteen_data[independent_variable]
    X = sm.add_constant(
        reg_nineteen_data.loc[
            :,
            regression_variables_5
            + demographics
            + household
            + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_nineteen = sm.WLS(y, X, weights=reg_nineteen_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_nineteen_data["village"]}
    )

    # COLUMN 4

    reg_twenty_data = df_reg_3.loc[
        :,
        independent_variable
        + regression_variables_5
        + demographics
        + household
        + sample_weights
        + village
        + dist
        + ["female_bihar", "bihar", "female"],
    ].dropna(how="any")

    y = reg_twenty_data[independent_variable]
    X = sm.add_constant(
        reg_twenty_data.loc[
            :,
            regression_variables_5
            + demographics
            + household
            + village
            + dist
            + ["female_bihar", "bihar", "female"],
        ]
    )

    reg_twenty = sm.WLS(y, X, weights=reg_twenty_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_twenty_data["village"]}
    )

    # * TABLE - 6 - PANEL - C - ROW - 2 *
    # * Quadruple-Difference (w.r.t. Jharkhand and Distance)*
    # * The treatment group is 13, 14 and 15 while the control group is 16 - long-distance dummy (greater than 3km) *

    # COLUMN 1
    y = df_reg_3.loc[:, independent_variable]
    X = sm.add_constant(
        df_reg_3.loc[
            :,
            regression_variables_5
            + regression_variables_6
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )
    reg_twenone = sm.WLS(y, X, weights=df_reg_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_3["village"]}
    )

    # COLUMN 2

    X = sm.add_constant(
        df_reg_3.loc[
            :,
            regression_variables_5
            + regression_variables_6
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ]
            + demographics,
        ]
    )

    reg_twentwo = sm.WLS(y, X, weights=df_reg_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_reg_3["village"]}
    )

    # COLUMN 3

    reg_twenthree_data = df_reg_3.loc[
        :,
        independent_variable
        + regression_variables_5
        + regression_variables_6
        + demographics
        + household
        + sample_weights
        + [
            "female_bihar",
            "bihar",
            "female",
            "longdist",
            "bihar_longdist",
            "female_longdist",
            "female_bihar_longdist",
        ],
    ].dropna(how="any")

    y = reg_twenthree_data[independent_variable]
    X = sm.add_constant(
        reg_twenthree_data.loc[
            :,
            regression_variables_5
            + regression_variables_6
            + demographics
            + household
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )

    reg_twenthree = sm.WLS(y, X, weights=reg_twenthree_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_twenthree_data["village"]}
    )

    # COLUMN 4

    reg_twenfour_data = df_reg_3.loc[
        :,
        independent_variable
        + regression_variables_5
        + regression_variables_6
        + demographics
        + household
        + sample_weights
        + village
        + dist
        + [
            "female_bihar",
            "bihar",
            "female",
            "longdist",
            "bihar_longdist",
            "female_longdist",
            "female_bihar_longdist",
        ],
    ].dropna(how="any")

    y = reg_twenfour_data[independent_variable]
    X = sm.add_constant(
        reg_twenfour_data.loc[
            :,
            regression_variables_5
            + regression_variables_6
            + demographics
            + household
            + village
            + dist
            + [
                "female_bihar",
                "bihar",
                "female",
                "longdist",
                "bihar_longdist",
                "female_longdist",
                "female_bihar_longdist",
            ],
        ]
    )
    reg_twenfour = sm.WLS(y, X, weights=reg_twenfour_data["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_twenfour_data["village"]}
    )

    table_6_str = md(
        "|||||| \n"
        "|:---|:---:|:---:|:---:|:---:| \n"
        "|<td colspan=5>Dependent variable: Enrolled in or completed grade 9| \n"
        "||(1)|(2)|(3)|(4)| \n"
        "|*Panel A. Treatment = 13, 14, 15; control = 16, 17*||||| \n"
        f"|Treat × female × Bihar|{round(reg_one.params['treat2_female_bihar'], 3)}|{round(reg_two.params['treat2_female_bihar'], 3)}|{round(reg_three.params['treat2_female_bihar'], 3)}|{round(reg_four.params['treat2_female_bihar'], 3)}| \n"
        f"||({round(reg_one.bse['treat2_female_bihar'], 3)})|({round(reg_two.bse['treat2_female_bihar'], 3)})|({round(reg_three.bse['treat2_female_bihar'], 3)})|({round(reg_four.bse['treat2_female_bihar'], 3)})| \n"
        f"|Treat × female × Bihar × long distance indicator|{round(reg_five.params['treat2_female_bihar_longdist'], 3)}|{round(reg_six.params['treat2_female_bihar_longdist'], 3)}|{round(reg_seven.params['treat2_female_bihar_longdist'], 3)}|{round(reg_eight.params['treat2_female_bihar_longdist'], 3)}| \n"
        f"||({round(reg_five.bse['treat2_female_bihar_longdist'], 3)})|({round(reg_six.bse['treat2_female_bihar_longdist'], 3)})|({round(reg_seven.bse['treat2_female_bihar_longdist'], 3)})|({round(reg_eight.bse['treat2_female_bihar_longdist'], 3)})| \n"
        f"|Observations|{int(reg_five.nobs)}|{int(reg_six.nobs)}|{int(reg_seven.nobs)}|{int(reg_eight.nobs)}| \n"
        "|*Panel B. Treatment = 14, 15; control = 16*||||| \n"
        f"|Treat × female × Bihar|{round(reg_nine.params['treat3_female_bihar'], 3)}|{round(reg_ten.params['treat3_female_bihar'], 3)}|{round(reg_eleven.params['treat3_female_bihar'], 3)}|{round(reg_twelve.params['treat3_female_bihar'], 3)}| \n"
        f"||({round(reg_nine.bse['treat3_female_bihar'], 3)})|({round(reg_ten.bse['treat3_female_bihar'], 3)})|({round(reg_eleven.bse['treat3_female_bihar'], 3)})|({round(reg_twelve.bse['treat3_female_bihar'], 3)})| \n"
        f"|Treat × female × Bihar × long distance indicator|{round(reg_thirteen.params['treat3_female_bihar_longdist'], 3)}|{round(reg_fourteen.params['treat3_female_bihar_longdist'], 3)}|{round(reg_fifteen.params['treat3_female_bihar_longdist'], 3)}|{round(reg_sixteen.params['treat3_female_bihar_longdist'], 3)}| \n"
        f"||({round(reg_thirteen.bse['treat3_female_bihar_longdist'], 3)})|({round(reg_fourteen.bse['treat3_female_bihar_longdist'], 3)})|({round(reg_fifteen.bse['treat3_female_bihar_longdist'], 3)})|({round(reg_sixteen.bse['treat3_female_bihar_longdist'], 3)})| \n"
        f"|Observations|{int(reg_nine.nobs)}|{int(reg_ten.nobs)}|{int(reg_eleven.nobs)}|{int(reg_twelve.nobs)}| \n"
        "|*Panel C. Treatment = 13, 14, 15; control = 16*||||| \n"
        f"|Treat × female × Bihar|{round(reg_seventeen.params['treat4_female_bihar'], 3)}|{round(reg_eighteen.params['treat4_female_bihar'], 3)}|{round(reg_nineteen.params['treat4_female_bihar'], 3)}|{round(reg_twenty.params['treat4_female_bihar'], 3)}| \n"
        f"||({round(reg_seventeen.bse['treat4_female_bihar'], 3)})|({round(reg_eighteen.bse['treat4_female_bihar'], 3)})|({round(reg_nineteen.bse['treat4_female_bihar'], 3)})|({round(reg_twenty.bse['treat4_female_bihar'], 3)})| \n"
        f"|Treat × female × Bihar × long distance indicator|{round(reg_twenone.params['treat4_female_bihar_longdist'], 3)}|{round(reg_twentwo.params['treat4_female_bihar_longdist'], 3)}|{round(reg_twenthree.params['treat4_female_bihar_longdist'], 3)}|{round(reg_twenfour.params['treat4_female_bihar_longdist'], 3)}| \n"
        f"||({round(reg_twenone.bse['treat4_female_bihar_longdist'], 3)})|({round(reg_twentwo.bse['treat4_female_bihar_longdist'], 3)})|({round(reg_twenthree.bse['treat4_female_bihar_longdist'], 3)})|({round(reg_twenfour.bse['treat4_female_bihar_longdist'], 3)})| \n"
        f"|Observations|{int(reg_seventeen.nobs)}|{int(reg_eighteen.nobs)}|{int(reg_nineteen.nobs)}|{int(reg_twenty.nobs)}| \n"
        "|Demographic controls|No|Yes|Yes|Yes| \n"
        "|HH socioeconomic controls|No|No|Yes|Yes| \n"
        "|Village level controls|No|No|No|Yes| \n"
        "<td colspan=5>Notes: The first row in each panel presents the triple difference coefficients analogous to the"
        + "first row in Table 2 but  with  different  cohorts  in  the  estimation  sample  and  different  definitions"
        + "of  treatment  and  control  cohorts  (as indicated in the panel title). The second row in each panel presents"
        + "the quadruple difference coefficients analogous to the first row in Table 3 with the same modified definitions "
        + "of treatment and control cohorts. The controls in the four columns are identical to those in Tables 2 and 3. "
        + "Standard errors, clustered by village ID, are in parentheses. \n"
    )

    return table_6_str


def table_7(dlhs_reg_data):
    df = dlhs_reg_data.copy()
    df = df.loc[np.isfinite(df["bihar"])]

    demographics = ["sc", "st", "obc", "hindu", "muslim"]
    household = ["hhheadschool", "hhheadmale", "land", "bpl", "media", "electricity"]

    village = ["middle", "bank", "postoff", "lcurrpop"]

    dist = ["busdist", "towndist", "railwaydist", "hqdist"]

    sample_weights = ["village", "hhwt"]

    all_vars = (
        demographics
        + household
        + village
        + dist
        + sample_weights
        + ["enrollment_secschool", "female", "female_bihar", "bihar"]
    )

    df_1 = df.loc[(df["bihar"] == 1) & (df["age"] == 13), all_vars]
    df_2 = df.loc[(df["bihar"] == 1) & (df["age"] == 14), all_vars]
    df_3 = df.loc[(df["bihar"] == 1) & (df["age"] == 15), all_vars]
    df_4 = df.loc[(df["bihar"] == 1) & (df["age"] == 16), all_vars]
    df_5 = df.loc[(df["bihar"] == 1) & (df["age"] == 17), all_vars]

    df_6 = df.loc[(df["bihar"] == 0) & (df["age"] == 13), all_vars]
    df_7 = df.loc[(df["bihar"] == 0) & (df["age"] == 14), all_vars]
    df_8 = df.loc[(df["bihar"] == 0) & (df["age"] == 15), all_vars]
    df_9 = df.loc[(df["bihar"] == 0) & (df["age"] == 16), all_vars]
    df_10 = df.loc[(df["bihar"] == 0) & (df["age"] == 17), all_vars]

    df_11 = df.loc[(df["age"] == 13), all_vars]
    df_12 = df.loc[(df["age"] == 14), all_vars]
    df_13 = df.loc[(df["age"] == 15), all_vars]
    df_14 = df.loc[(df["age"] == 16), all_vars]
    df_15 = df.loc[(df["age"] == 17), all_vars]

    df_16 = df.loc[(df["age"] == 13) & (df["longdist"] == 1), all_vars]
    df_17 = df.loc[(df["age"] == 14) & (df["longdist"] == 1), all_vars]
    df_18 = df.loc[(df["age"] == 15) & (df["longdist"] == 1), all_vars]
    df_19 = df.loc[(df["age"] == 16) & (df["longdist"] == 1), all_vars]
    df_20 = df.loc[(df["age"] == 17) & (df["longdist"] == 1), all_vars]

    ## Finding averages
    # PANEL A

    df_21 = df.loc[
        (df["age"] == 13) & (df["bihar"] == 1) & (df["female"] == 0), all_vars
    ]
    df_22 = df.loc[
        (df["age"] == 14) & (df["bihar"] == 1) & (df["female"] == 0), all_vars
    ]
    df_23 = df.loc[
        (df["age"] == 15) & (df["bihar"] == 1) & (df["female"] == 0), all_vars
    ]
    df_24 = df.loc[
        (df["age"] == 16) & (df["bihar"] == 1) & (df["female"] == 0), all_vars
    ]
    df_25 = df.loc[
        (df["age"] == 17) & (df["bihar"] == 1) & (df["female"] == 0), all_vars
    ]

    ## Finding averages
    # PANEL B

    df_26 = df.loc[
        (df["age"] == 13) & (df["bihar"] == 0) & (df["female"] == 0), all_vars
    ]
    df_27 = df.loc[
        (df["age"] == 14) & (df["bihar"] == 0) & (df["female"] == 0), all_vars
    ]
    df_28 = df.loc[
        (df["age"] == 15) & (df["bihar"] == 0) & (df["female"] == 0), all_vars
    ]
    df_29 = df.loc[
        (df["age"] == 16) & (df["bihar"] == 0) & (df["female"] == 0), all_vars
    ]
    df_30 = df.loc[
        (df["age"] == 17) & (df["bihar"] == 0) & (df["female"] == 0), all_vars
    ]

    # TABLE 7 - PANEL A

    reg_t7_1 = df_1.dropna(how="any")

    y = reg_t7_1["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_1.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_one = sm.WLS(y, X, weights=reg_t7_1["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_1["village"]}
    )

    reg_t7_2 = df_2.dropna(how="any")

    y = reg_t7_2["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_2.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_two = sm.WLS(y, X, weights=reg_t7_2["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_2["village"]}
    )

    reg_t7_3 = df_3.dropna(how="any")

    y = reg_t7_3["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_3.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_three = sm.WLS(y, X, weights=reg_t7_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_3["village"]}
    )

    reg_t7_4 = df_4.dropna(how="any")

    y = reg_t7_4["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_4.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_four = sm.WLS(y, X, weights=reg_t7_4["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_4["village"]}
    )

    reg_t7_5 = df_5.dropna(how="any")

    y = reg_t7_5["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_5.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_five = sm.WLS(y, X, weights=reg_t7_5["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_5["village"]}
    )

    # * WE START WITH SINGLE DIFFERENCE IN JHARKHAND, I.E. GIRLS VS. BOYS. WE RUN SEPERATE REGRESSION FOR AGE 13 THOUGH 17 *

    # * SINGLE DIFFERENCE W.R.T BOYS IN JHARKHAND *
    # * TABLE 7 - PANEL - B *

    reg_t7_6 = df_6.dropna(how="any")

    y = reg_t7_6["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_6.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_six = sm.WLS(y, X, weights=reg_t7_6["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_6["village"]}
    )

    reg_t7_7 = df_7.dropna(how="any")

    y = reg_t7_7["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_7.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_seven = sm.WLS(y, X, weights=reg_t7_7["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_7["village"]}
    )

    reg_t7_8 = df_8.dropna(how="any")

    y = reg_t7_8["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_8.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_eight = sm.WLS(y, X, weights=reg_t7_8["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_8["village"]}
    )

    reg_t7_9 = df_9.dropna(how="any")

    y = reg_t7_9["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_9.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_nine = sm.WLS(y, X, weights=reg_t7_9["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_9["village"]}
    )

    reg_t7_10 = df_10.dropna(how="any")

    y = reg_t7_10["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_10.loc[:, ["female"] + demographics + household + village + dist]
    )

    reg_ten = sm.WLS(y, X, weights=reg_t7_10["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_10["village"]}
    )

    # * NOW WE WILL DO DIFF-IN-DIFF (GIRLS VS. BOYS; BH VS. JH) *
    # * TABLE 7 - PANEL - C *

    reg_t7_11 = df_11.dropna(how="any")

    y = reg_t7_11["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_11.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_eleven = sm.WLS(y, X, weights=reg_t7_11["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_11["village"]}
    )

    reg_t7_12 = df_12.dropna(how="any")

    y = reg_t7_12["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_12.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_twelve = sm.WLS(y, X, weights=reg_t7_12["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_12["village"]}
    )

    reg_t7_13 = df_13.dropna(how="any")

    y = reg_t7_13["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_13.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_thirteen = sm.WLS(y, X, weights=reg_t7_13["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_13["village"]}
    )

    reg_t7_14 = df_14.dropna(how="any")

    y = reg_t7_14["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_14.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_fourteen = sm.WLS(y, X, weights=reg_t7_14["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_14["village"]}
    )

    reg_t7_15 = df_15.dropna(how="any")

    y = reg_t7_15["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_15.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_fifteen = sm.WLS(y, X, weights=reg_t7_15["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_15["village"]}
    )

    # * NOW WE WILL DO DIFF-IN-DIFF (GIRLS VS. BOYS; BH VS. JH) FOR SHORT DISTANCE, I.E. IF SCHOOL IS GREATHER THAN 3 KMS *
    # * TABLE 7 - PANEL - D *

    reg_t7_16 = df_16.dropna(how="any")

    y = reg_t7_16["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_16.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_sixteen = sm.WLS(y, X, weights=reg_t7_16["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_16["village"]}
    )

    reg_t7_17 = df_17.dropna(how="any")

    y = reg_t7_17["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_17.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_seventeen = sm.WLS(y, X, weights=reg_t7_17["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_17["village"]}
    )

    reg_t7_18 = df_18.dropna(how="any")

    y = reg_t7_18["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_18.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_eighteen = sm.WLS(y, X, weights=reg_t7_18["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_18["village"]}
    )

    reg_t7_19 = df_19.dropna(how="any")

    y = reg_t7_19["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_19.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_nineteen = sm.WLS(y, X, weights=reg_t7_19["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_19["village"]}
    )

    reg_t7_20 = df_20.dropna(how="any")

    y = reg_t7_20["enrollment_secschool"]
    X = sm.add_constant(
        reg_t7_20.loc[
            :,
            ["female_bihar", "female", "bihar"]
            + demographics
            + household
            + village
            + dist,
        ]
    )

    reg_twenty = sm.WLS(y, X, weights=reg_t7_20["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_t7_20["village"]}
    )

    ## FINDING AVERAGES
    # PANEL A

    av_1 = df_21["enrollment_secschool"].mean()
    av_2 = df_22["enrollment_secschool"].mean()
    av_3 = df_23["enrollment_secschool"].mean()
    av_4 = df_24["enrollment_secschool"].mean()
    av_5 = df_25["enrollment_secschool"].mean()

    # PANEL B

    av_6 = df_26["enrollment_secschool"].mean()
    av_7 = df_27["enrollment_secschool"].mean()
    av_8 = df_28["enrollment_secschool"].mean()
    av_9 = df_29["enrollment_secschool"].mean()
    av_10 = df_30["enrollment_secschool"].mean()

    table_7_str = md(
        "||||||| \n"
        "|:---|:---:|:---:|:---:|:---:|:---:| \n"
        "|<td colspan=6>Dependent variable: Enrolled in or completed grade 9| \n"
        "||Age=13|Age=14|Age=15|Age=16|Age=17|"
        "||(1)|(2)|(3)|(4)|(5)| \n"
        "|*Panel A. Gender gap in secondary schooling in age (bihar)*|||||| \n"
        f"|Female|{round(reg_one.params['female'], 3)}|{round(reg_two.params['female'], 3)}|{round(reg_three.params['female'], 3)}|{round(reg_four.params['female'], 3)}|{round(reg_five.params['female'], 3)}| \n"
        f"||({round(reg_one.bse['female'], 3)})|({round(reg_two.bse['female'], 3)})|({round(reg_three.bse['female'], 3)})|({round(reg_four.bse['female'], 3)})|({round(reg_five.bse['female'], 3)})| \n"
        f"|Averages for boys|{round(float(av_1), 3)}|{round(float(av_2), 3)}|{round(float(av_3), 3)}|{round(float(av_4), 3)}|{round(float(av_5), 3)}| \n"
        f"|Observations|{int(reg_one.nobs)}|{int(reg_two.nobs)}|{int(reg_three.nobs)}|{int(reg_four.nobs)}|{int(reg_five.nobs)}| \n"
        "|*Panel B. Gender gap in secondary schooling in age (Jharkhand)*|||||| \n"
        f"|Female|{round(reg_six.params['female'], 3)}|{round(reg_seven.params['female'], 3)}|{round(reg_eight.params['female'], 3)}|{round(reg_nine.params['female'], 3)}|{round(reg_ten.params['female'], 3)}| \n"
        f"||({round(reg_six.bse['female'], 3)})|({round(reg_seven.bse['female'], 3)})|({round(reg_eight.bse['female'], 3)})|({round(reg_nine.bse['female'], 3)})|({round(reg_ten.bse['female'], 3)})| \n"
        f"|Averages for boys|{round(float(av_6), 3)}|{round(float(av_7), 3)}|{round(float(av_8), 3)}|{round(float(av_9), 3)}|{round(float(av_10), 3)}| \n"
        f"|Observations|{int(reg_six.nobs)}|{int(reg_seven.nobs)}|{int(reg_eight.nobs)}|{int(reg_nine.nobs)}|{int(reg_ten.nobs)}| \n"
        "|*Panel C. Differential gender gap in secondary schooling by age (Bihar versus Jharkhand)*|||||| \n"
        f"|Female x Bihar|{round(reg_eleven.params['female_bihar'], 3)}|{round(reg_twelve.params['female_bihar'], 3)}|{round(reg_thirteen.params['female_bihar'], 3)}|{round(reg_fourteen.params['female_bihar'], 3)}|{round(reg_fifteen.params['female_bihar'], 3)}| \n"
        f"||({round(reg_eleven.bse['female_bihar'], 3)})|({round(reg_twelve.bse['female_bihar'], 3)})|({round(reg_thirteen.bse['female_bihar'], 3)})|({round(reg_fourteen.bse['female_bihar'], 3)})|({round(reg_fifteen.bse['female_bihar'], 3)})| \n"
        f"|Observations|{int(reg_eleven.nobs)}|{int(reg_twelve.nobs)}|{int(reg_thirteen.nobs)}|{int(reg_fourteen.nobs)}|{int(reg_fifteen.nobs)}| \n"
        "|*Panel D. Differential gender gap in secondary schooling by age (Bihar versus Jharkhand)-restricted to villages that are 3 km or farther away from a secondary school*|||||| \n"
        f"|Female x Bihar|{round(reg_sixteen.params['female_bihar'], 3)}|{round(reg_seventeen.params['female_bihar'], 3)}|{round(reg_eighteen.params['female_bihar'], 3)}|{round(reg_nineteen.params['female_bihar'], 3)}|{round(reg_twenty.params['female_bihar'], 3)}| \n"
        f"||({round(reg_sixteen.bse['female_bihar'], 3)})|({round(reg_seventeen.bse['female_bihar'], 3)})|({round(reg_eighteen.bse['female_bihar'], 3)})|({round(reg_nineteen.bse['female_bihar'], 3)})|({round(reg_twenty.bse['female_bihar'], 3)})| \n"
        f"|Observations|{int(reg_sixteen.nobs)}|{int(reg_seventeen.nobs)}|{int(reg_eighteen.nobs)}|{int(reg_nineteen.nobs)}|{int(reg_twenty.nobs)}| \n"
        "|Demographic controls|Yes|Yes|Yes|Yes|Yes| \n"
        "|HH socioeconomic controls|Yes|Yes|Yes|Yes|Yes| \n"
        "|Village level controls|Yes|Yes|Yes|Yes|Yes| \n"
        "<td colspan=5>Notes: The demographic, socioeconomic, and village controls are the same as those shown in Table 2 and online Appendix Table A.1, and are included in all regressions (to enable comparison with Table 2, column 4). Standard errors, clustered by village ID, are in parentheses."
    )
    return table_7_str


def table_8(dlhs_reg_data):
    df = dlhs_reg_data.copy()
    df.rename(
        {"enrollment_middleschool": "Enrolled in or completed grade 8"},
        axis=1,
        inplace=True,
    )

    # regression 2 data
    independent_variable = ["Enrolled in or completed grade 8"]

    regression_variables_1 = [
        "treat5_female_bihar",
        "treat5_female",
        "treat5_bihar",
        "female_bihar",
        "treat5",
        "female",
        "bihar",
    ]

    demographics = ["sc", "st", "obc", "hindu", "muslim"]

    household = ["hhheadschool", "hhheadmale", "land", "bpl", "media", "electricity"]

    village = ["middle", "bank", "postoff", "lcurrpop"]

    dist = ["busdist", "towndist", "railwaydist", "hqdist"]

    sample_weights = ["village", "hhwt"]

    all_vars = (
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + village
        + dist
        + sample_weights
    )

    df_table_8 = df.loc[
        (df["bihar"] == df["bihar"]) & (df["treat5"] == df["treat5"]),
        all_vars,
    ]

    # /* Table 8 */

    # * Triple Difference (w.r.t. Jharkhand) *
    # * The treatment group is 13 and 14 while the control group is 15 and 16 *

    # COLUMN 1

    y = df_table_8.loc[:, independent_variable]
    X = sm.add_constant(df_table_8.loc[:, regression_variables_1])

    reg_8_1 = sm.WLS(y, X, weights=df_table_8["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_table_8["village"]}
    )

    # COLUMN 2
    X = sm.add_constant(df_table_8.loc[:, regression_variables_1 + demographics])

    reg_8_2 = sm.WLS(y, X, weights=df_table_8["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": df_table_8["village"]}
    )

    reg_8_3 = df_table_8.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights,
    ].dropna(how="any")

    y = reg_8_3[independent_variable]
    X = sm.add_constant(
        reg_8_3.loc[:, regression_variables_1 + demographics + household]
    )
    # COLUMN 3
    reg_8_3 = sm.WLS(y, X, weights=reg_8_3["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_8_3["village"]}
    )

    # COLUMN 4
    reg_8_4 = df_table_8.loc[
        :,
        independent_variable
        + regression_variables_1
        + demographics
        + household
        + sample_weights
        + village
        + dist,
    ].dropna(how="any")

    y = reg_8_4[independent_variable]
    X = sm.add_constant(
        reg_8_4.loc[
            :, regression_variables_1 + demographics + household + village + dist
        ]
    )

    reg_8_4 = sm.WLS(y, X, weights=reg_8_4["hhwt"]).fit(
        cov_type="cluster", cov_kwds={"groups": reg_8_4["village"]}
    )
    reg_8_4.summary()

    ## Table creation with stargazer package
    stargazer = Stargazer(
        [
            reg_8_1,
            reg_8_2,
            reg_8_3,
            reg_8_4,
        ]
    )
    stargazer.title(
        "Table 8 -- Triple Difference (DDD) Estimate of the Impact of Being Exposed to the Cycle Program On Girl's Enrollment in Eighth Grade (placebo test)"
    )
    stargazer.custom_columns(
        "Treatment group = age 13 and 14 \
       Control group = age 15 and 16"
    )

    stargazer.covariate_order(
        [
            "treat5_female_bihar",
            "treat5_female",
            "treat5_bihar",
            "female_bihar",
            "treat5",
            "female",
            "bihar",
            "const",
        ]
    )
    stargazer.rename_covariates(
        {
            "treat5_female_bihar": "Treat x female x Bihar",
            "treat5_female": "Treat x female",
            "treat5_bihar": "Treat x Bihar",
            "female_bihar": "Female x Bihar",
            "treat5": "Treat",
            "female": "Female",
            "bihar": "Bihar",
            "const": "Constant",
        }
    )
    stargazer.custom_note_label(
        "Notes: Unlike Table 2 that uses an estimation sample of household residents aged"
        + "14–17, this table uses household residents aged 13–16 as the estimation sample. "
        + "The demographic, socioeconomic, and village controls are the same as those shown "
        + "in Table 2 and online Appendix Table A.1. Standard errors, clustered by village ID,"
        + "are in parentheses."
    )

    return stargazer


table_sum = md(
    "|||||| \n"
    "|:---|:---:|:---:|:---:|:---:| \n"
    "|**Summary Statistics (estimation sample)**||||| \n"
    "||Bihar|Jharkhand| \n"
    "|**Panel A. Dependent variable**||||| \n"
    f"|Enrolled in or completed grade 9 (Among 14-17 year olds)|{0.31}|{0.34}| \n"
    f"||(0.46)|(0.47)| \n"
    "|**PANEL B: Key independent variables**||||| \n"
    f"|Treatment group = Child age 14 & 15 (Among 14-17 year olds)|{0.54}|{0.59}| \n"
    f"||(0.50)|(0.49)| \n"
    f"|Female|{0.49}|{0.47}| \n"
    f"||(0.50)|(0.50)| \n"
    "|**PANEL C: Demographic Characteristics**||||| \n"
    f"|Social group: Scheduled caste|{0.19}|{0.14}| \n"
    f"||(0.39)|(0.34)| \n"
    f"|Social group: Scheduled tribes|{0.02}|{0.36}| \n"
    f"||(0.15)|(0.48)| \n"
    f"|Social group: Other backward caste|{0.59}|{0.42}| \n"
    f"||(0.49)|(0.49)| \n"
    f"|Social group: Hindu|{(0.85)}|{0.65}| \n"
    f"||(0.36)|(0.48)| \n"
    f"|Social group: Muslim|{(0.15)}|{0.12}| \n"
    f"||(0.36)|(0.32)| \n"
    "|**PANEL D: Demographic Characteristics**||||| \n"
    f"|Household head years of schooling|{4.32}|{3.94}| \n"
    f"||(5.03)|(4.43)| \n"
    f"|Household head Male|{0.86}|{0.95}| \n"
    f"||(0.35)|(0.21)| \n"
    f"|Land (<5 acres = marginal farmer)|{0.95}|{0.93}| \n"
    f"||(0.22)|(0.25)| \n"
    f"|Below poverty line|{(0.29)}|{0.40}| \n"
    f"||(0.45)|(0.49)| \n"
    f"|Household owns TV/Radio|{(0.272)}|{0.31}| \n"
    f"||(0.45)|(0.46)| \n"
    f"|Household access to electricity|{(0.20)}|{0.26}| \n"
    f"||(0.40)|(0.44)| \n"
    "|**PANEL E: Village Characteristics**||||| \n"
    f"|Primary school in village|{0.88}|{0.89}| \n"
    f"||(0.32)|(0.31)| \n"
    f"|Middle school in village|{0.47}|{0.54}| \n"
    f"||(0.50)|(0.50)| \n"
    f"|Secondary school in village|{0.11}|{0.07}| \n"
    f"||(0.32)|(0.26)| \n"
    f"|Bank in village|{(0.10)}|{0.06}| \n"
    f"||(0.30)|(0.24)| \n"
    f"|Post office in village|{(0.32)}|{0.21}| \n"
    f"||(0.47)|(0.41)| \n"
    f"|Distance to bus station (km)|{(7.35)}|{12.15}| \n"
    f"||(9.94)|(12.81)| \n"
    f"|Distance to nearest town (km)|{(14.00)}|{17.65}| \n"
    f"||(13.94)|(15.46)| \n"
    f"|Distance to railway station (km)|{(18.21)}|{33.96}| \n"
    f"||(42.12)|(30.64)| \n"
    f"|Distance to district headquarter (km)|{(32.94)}|{39.39}| \n"
    f"||(37.26)|(24.51)| \n"
    f"|Log (Village current population)|{(7.79)}|{6.87}| \n"
    f"||(1.17)|(0.83)| \n"
    f"|Observations|18,453|11,842| \n"
)


def create_table1(regression_data):
    """
    Creates Table 1.
    """
    variables = regression_data[
        [
            "enrollment_secschool",
            "treat1_female_bihar",
            "treat1_female",
            "treat1_bihar",
            "female_bihar",
            "treat1",
            "female",
            "bihar",
            "treat1_female_bihar_longdist",
            "treat1_female_longdist",
            "female_bihar_longdist",
            "treat1_bihar_longdist",
            "treat1_longdist",
            "female_longdist",
            "bihar_longdist",
            "longdist",
        ]
    ]

    table1 = pd.DataFrame()
    table1["Minimum"] = variables.min()
    table1["Mean"] = variables.mean()
    table1["Median"] = variables.median()
    table1["Standard Deviation"] = variables.std()
    table1["Maximum"] = variables.max()
    table1 = table1.astype(float).round(2)
    table1["Description"] = [
        "Indicator variable for enrolled in or completed grade 9 (secondary school)",
        "Females in Bihar in age group age 14-15 years old",
        "Females aged 14-15 years old",
        "Aged 14-15 years old in Bihar",
        "Females in Bihar",
        "Main treatment - aged 14-15 years old",
        "Female",
        "Bihar",
        "Females in Bihar in age group age 14-15 years old located 3km or further away from secondary school",
        "Females aged 14-15 years old located 3km or further away from secondary school",
        "Females in Bihar located 3km or further away from secondary school",
        "Age group 14-15 years old in Bihar located 3km or further away from secondary school",
        "Age group 14-15 years old located 3km or further away from secondary school",
        "Females located 3km or further away from secondary school",
        "Bihar located 3km or further away from secondary school",
        "located 3km or further away from secondary school",
    ]

    table1.loc[0:1, "Type"] = "Outcomes"
    table1.loc[1:, "Type"] = "Covariates"

    return table1


## Extension - Placebo test


def gen_post_1(value):
    if value in [2006, 2007]:
        return 1
    elif value in [2004, 2005]:
        return 0
    else:
        return np.nan


def table_9(exam_data):
    df = exam_data.copy()
    df = df.loc[df["year"] != 2008].copy()
    df["post"] = df["year"].apply(gen_post_1)
    df_mean = df.groupby(["school_code", "statecode", "post", "gender", "statename"])[
        ["appear_tot", "pass_tot", "district_code"]
    ].apply(lambda x: x.mean(skipna=True))
    df_mean.reset_index(inplace=True)
    df_mean["treat"] = df_mean["statecode"].apply(lambda x: 1 if x == 1 else 0)
    df_mean["male"] = df_mean["gender"].apply(lambda x: 1 if x == 1 else 0)
    df_mean["female"] = df_mean["gender"].apply(lambda x: 1 if x == 2 else 0)
    df_mean["bh_post"] = df_mean["treat"] * df_mean["post"]
    df_mean["female_post"] = df_mean["female"] * df_mean["post"]
    df_mean["female_treat"] = df_mean["female"] * df_mean["treat"]
    df_mean["triple"] = df_mean["treat"] * df_mean["post"] * df_mean["female"]
    df_mean["lappear"] = np.log(df_mean["appear_tot"])
    df_mean["lpass"] = np.log(df_mean["pass_tot"])
    df_1 = df_mean.copy()
    gp = df_1.groupby(["school_code", "gender"])["lappear"]
    df_1["sch_gender_prepost_appear"] = gp.transform(lambda x: np.isfinite(x).sum())
    reg_df = df_1.loc[df_1["sch_gender_prepost_appear"] == 2]
    X = sm.add_constant(
        reg_df.loc[
            :,
            [
                "triple",
                "female_treat",
                "bh_post",
                "female_post",
                "female",
                "treat",
                "post",
            ],
        ]
    )
    y = reg_df["lappear"]
    model = sm.OLS(y, X)
    res_1 = model.fit(cov_type="HC1")

    gp = df_1.groupby(["school_code", "gender"])["lpass"]
    df_1["sch_gender_prepost_pass"] = gp.transform(lambda x: np.isfinite(x).sum())
    reg_df = df_1.loc[df_1["sch_gender_prepost_pass"] == 2]
    X = sm.add_constant(
        reg_df.loc[
            :,
            [
                "triple",
                "female_treat",
                "bh_post",
                "female_post",
                "female",
                "treat",
                "post",
            ],
        ]
    )
    y = reg_df["lpass"]
    model = sm.OLS(y, X)
    res_2 = model.fit(cov_type="HC1")

    ## Table creation with stargazer package
    stargazer = Stargazer([res_1, res_2])
    stargazer.title("Table 9")
    # stargazer.custom_columns(
    #    "Triple difference (DDD) estimate of impact of exposure to cycle program"
    # )

    stargazer.custom_columns(
        [
            "log (number of candidates who appeared for the 10th grade exam)",
            "log (number of candidates who passed the 10th grade exam)",
        ],
        [1, 1],
    )
    stargazer.covariate_order(
        [
            "triple",
            "female_treat",
            "bh_post",
            "female_post",
            "female",
            "treat",
            "post",
            "const",
        ]
    )
    stargazer.rename_covariates(
        {
            "triple": "Bihar x female x post",
            "female_treat": "Female x Bihar",
            "bh_post": "Bihar x post",
            "female_post": "Female x post",
            "female": "Female",
            "treat": "Bihar",
            "post": "Post",
            "const": "Constant",
        }
    )
    stargazer.custom_note_label(
        "Notes: The analysis uses data on the secondary school certificate (SSC) examination (10th standard board exam"
        + "records) from the State Examination Board Authorities in Bihar and Jharkhand for the years 2004–2010. The pre-period"
        + "is defined as the school years ending in 2004 to 2007, and the post-period is defined as the school years ending in 2009"
        + "and 2010. We first calculate the average number of students who appeared in and passed the exams for each school by gender"
        + "over the four years in the pre-period and the two years in the post-period, and then take the log of this average to generate"
        + "one observation for each school by gender in the “pre” and “post” periods. The sample is restricted to schools where both pre-"
        + "and post-data exist for a given gender. We calculate standard errors both with and without clustering, but find that clustering lowers the standard errors. We therefore report the more conservative unclustered standard errors."
    )
    return stargazer


def table_5_panel_a(dlhs_reg_data):
    df = dlhs_reg_data.loc[(np.isfinite(dlhs_reg_data['bihar'])) & dlhs_reg_data['age'].between(12, 18, inclusive=True)] 
    panel_a = df.loc[df['currgrade'] == 9].copy()
    total_n = panel_a.groupby(['female'])['age'].value_counts(False).to_frame('freq')
    total_p = panel_a.groupby(['female'])['age'].value_counts(True).to_frame('per')
    total = total_n.merge(total_p, left_index=True, right_index=True)
    total.loc[total.index.get_level_values('female') == 1].sum()
    total['state'] = 'total'
    total = total.reset_index()
    states_n = panel_a.groupby(['female', 'state'])['age'].value_counts(False).to_frame('freq')
    states_p = panel_a.groupby(['female', 'state'])['age'].value_counts(True).to_frame('per')
    states = states_n.merge(states_p, left_index=True, right_index=True)
    states = states.reset_index()
    panel_a_table_five = pd.concat([states, total])
    panel_a_table_five = panel_a_table_five.reset_index(drop=True).set_index(['female', 'state', 'age'])
    table_five_a = panel_a_table_five.sort_index(level=['female', 'state', 'age'])

    return table_five_a

def table_5_panel_b(dlhs_reg_data):
    df_2 = dlhs_reg_data.loc[(np.isfinite(dlhs_reg_data['bihar'])) & dlhs_reg_data['age'].between(13, 17, inclusive=True)] 
    panel_b = df_2.copy()
    total_n = panel_b.groupby(['female'])['age'].value_counts(False).to_frame('freq')
    total_p = panel_b.groupby(['female'])['age'].value_counts(True).to_frame('per')
    total = total_n.merge(total_p, left_index=True, right_index=True)
    total.loc[total.index.get_level_values('female') == 1].sum()
    total['state'] = 'total'
    total = total.reset_index()
    states_n = panel_b.groupby(['female', 'state'])['age'].value_counts(False).to_frame('freq')
    states_p = panel_b.groupby(['female', 'state'])['age'].value_counts(True).to_frame('per')
    states = states_n.merge(states_p, left_index=True, right_index=True)
    states = states.reset_index()
    panel_b_table_five = pd.concat([states, total])
    panel_b_table_five = panel_b_table_five.reset_index(drop=True).set_index(['female', 'state', 'age'])
    table_five_b = panel_b_table_five.sort_index(level=['female', 'state', 'age'])

    return table_five_b







