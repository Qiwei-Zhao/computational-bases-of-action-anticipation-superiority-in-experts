
library("bruceR") # load bruceR package


set.wd() # set working directory as the folder where the script is located
dat <- import("data/data for statistics.csv") # import data


# Task 1: NDs Action Anticipation #####

TTEST(dat, y = "TASK1_ACCURACY", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK1_RESPONSE_TIME", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK1_READOUT", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK1_OVERLAP", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK1_ALIGNMENT", x = "GROUP", factor.rev = "FALSE")

dat %>% filter(GROUP == "1") %>% TTEST(y = "TASK1_ACCURACY", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "TASK1_ACCURACY", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "1") %>% TTEST(y = "TASK1_READOUT", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "TASK1_READOUT", test.value = 0.5, factor.rev = "FALSE")

# Correlation Analysis 
dat %>% filter(GROUP == "1") %>%select(TASK1_ACCURACY, TASK1_READOUT, TASK1_OVERLAP, TASK1_ALIGNMENT, Training_Year) %>% Corr()
dat %>% filter(GROUP == "2") %>%select(TASK1_ACCURACY, TASK1_READOUT, TASK1_OVERLAP, TASK1_ALIGNMENT) %>% Corr()


# Task 2. PLDs Action Anticipation ####

TTEST(dat, y = "TASK2_ACCURACY", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK2_RESPONSE_TIME", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK2_READOUT", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK2_OVERLAP", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "TASK2_ALIGNMENT", x = "GROUP", factor.rev = "FALSE")

# one-sample t-test comparing with 0.5
dat %>% filter(GROUP == "1") %>% TTEST(y = "TASK2_ACCURACY", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "TASK2_ACCURACY", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "1") %>% TTEST(y = "TASK2_READOUT", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "TASK2_READOUT", test.value = 0.5, factor.rev = "FALSE")

# PLDs
dat %>% filter(GROUP == "1") %>%select(TASK2_ACCURACY, TASK2_READOUT, TASK2_OVERLAP, TASK2_ALIGNMENT, Training_Year) %>% Corr()
dat %>% filter(GROUP == "2") %>%select(TASK2_ACCURACY, TASK2_READOUT, TASK2_OVERLAP, TASK2_ALIGNMENT) %>% Corr()


# 3. Cross-Task Analysis #####


# Cross Alignment
TTEST(dat, y = "CROSS_OVERLAP", x = "GROUP", factor.rev = "FALSE")
TTEST(dat, y = "CROSS_ALIGNMENT", x = "GROUP", factor.rev = "FALSE")

# one-sample t-test comparing with 0.5
dat %>% filter(GROUP == "1") %>% TTEST(y = "CROSS_OVERLAP", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "CROSS_OVERLAP", test.value = 0.5, factor.rev = "FALSE")
dat %>% filter(GROUP == "1") %>% TTEST(y = "CROSS_ALIGNMENT", test.value = 0, factor.rev = "FALSE")
dat %>% filter(GROUP == "2") %>% TTEST(y = "CROSS_ALIGNMENT", test.value = 0, factor.rev = "FALSE")

# Correlation between Noraml Videos and PLDs
dat %>% filter(GROUP == "1") %>%select(TASK1_ACCURACY, TASK1_RESPONSE_TIME, TASK1_READOUT, TASK1_OVERLAP, TASK1_ALIGNMENT, TASK2_ACCURACY, TASK2_RESPONSE_TIME, TASK2_READOUT, TASK2_OVERLAP, TASK2_ALIGNMENT) %>% Corr()
dat %>% filter(GROUP == "2") %>%select(TASK1_ACCURACY, TASK1_RESPONSE_TIME, TASK1_READOUT, TASK1_OVERLAP, TASK1_ALIGNMENT, TASK2_ACCURACY, TASK2_RESPONSE_TIME, TASK2_READOUT, TASK2_OVERLAP, TASK2_ALIGNMENT) %>% Corr()
