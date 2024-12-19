# Nuisance Parameters
- Ensemble level parameters can be specificed in any or all experiments but must share the same name and value
- Experiment level parameters affecting a particular key (e.g. nin or brn) should take the form key_experiment_name
- If key starts with or is "flux" it will be applied to flux
- One value means normal distribution, array of two values means uinform:
        {"a": sigma_a,
        "b": [lower_b, upper_b],}