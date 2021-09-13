import os

# just add all the things you want to run, one after another here.
# Will probably stop if one of them crashes though

#os.system("python train.py experiment_template.cfg")
#os.system("python train.py experiment_template.cfg")

strings = ["python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_1_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x120_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x120v2_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x60_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x60v2_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x180_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x120+3x60_{}",
            "python show_loss_landscape.py ./output/depth/spherical_mapping_obj10_0+3x120+3x60+3x180_{}",]

for s in strings:
    for i in [0, 10, 100, 500, 1000]:
        os.system(s.format(i))
