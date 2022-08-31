import pyffe
from pyffe.models import mAlexNet, AlexNet

dataset_prefix = 'splits'

PKLot = pyffe.Dataset(dataset_prefix + '/PKLot')
CNRParkAB = pyffe.Dataset(dataset_prefix + '/CNRParkAB')
CNRExt = pyffe.Dataset(dataset_prefix + '/CNRPark-EXT')
Combined = pyffe.Dataset(dataset_prefix + '/Combined')
Custom = pyffe.Dataset(dataset_prefix + '/Custom')
Custom_GIST_220712 = pyffe.Dataset(dataset_prefix + '/Custom_GIST_220712')
Custom_Bluecom_220718 = pyffe.Dataset(dataset_prefix + '/Custom_Bluecom_220718')
Custom_Bluecom_220801 = pyffe.Dataset(dataset_prefix + '/Custom_Bluecom_220801')
Custom_Bluecom_220802 = pyffe.Dataset(dataset_prefix + '/Custom_Bluecom_220802')
Custom_Bluecom_Total = pyffe.Dataset(dataset_prefix + '/Custom_Bluecom_Total')
Custom_141753_001 = pyffe.Dataset(dataset_prefix + '/141753_001')
Custom_dji = pyffe.Dataset(dataset_prefix + '/Custom_dji')
Custom_insta = pyffe.Dataset(dataset_prefix + '/Custom_insta')
Custom_Paper = pyffe.Dataset(dataset_prefix + '/Custom_Paper')

input_format = pyffe.InputFormat(
    new_width=256,
    new_height=256,
    crop_size=224,
    scale=1. / 256,
    mirror=True
)

model = mAlexNet(input_format, num_output=2, batch_sizes=[64, 100])
bigmodel = AlexNet(input_format, num_output=2, batch_sizes=[64, 50])

solver = pyffe.Solver(
    base_lr=0.0008,
    train_epochs=6,
    lr_policy="step",
    gamma=0.75,
    stepsize_epochs=2,
    val_interval_epochs=0.15,
    val_epochs=0.1,
    display_per_epoch=30,
    snapshot_interval_epochs=0.15,
)
exps = [
    # exp 1.1 and 1.2
    #pyffe.Experiment(model, solver, PKLot.train, val=[PKLot.val, CNRExt.val], test=[PKLot.test, CNRExt.test]),
    #pyffe.Experiment(bigmodel, solver, PKLot.train, val=[PKLot.val, CNRExt.val], test=[PKLot.test, CNRExt.test]),
    
    # # exp 2.1 and 2.2
    # pyffe.Experiment(model, solver, CNRParkAB.all, val=[CNRExt.val, PKLot.valCNRExt test=[CNRExt.test, PKLot.test]),
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[CNRExt.val, PKLot.valCNRExt test=[CNRExt.test, PKLot.test]),

    # # exp 3.1 and 3.2
    # pyffe.Experiment(model, solver, Combined.CNRParkAB_Ext_train, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),

    # # exp 4.1 and 4.2
    # pyffe.Experiment(model, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[CNRExt.val, PKLot.val], test=[CNRExt.test, PKLot.test] ),
        
    # # Inter-weather experiments
    # pyffe.Experiment(model, solver, CNRExt.sunny, val=[CNRExt.overcast, CNRExt.rainy, PKLot.val], test=[CNRExt.overcast, CNRExt.rainy, PKLot.test]),
    # pyffe.Experiment(model, solver, CNRExt.overcast, val=[CNRExt.sunny, CNRExt.rainy, PKLot.val], test=[CNRExt.sunny, CNRExt.rainy, PKLot.test]),
    # pyffe.Experiment(model, solver, CNRExt.rainy, val=[CNRExt.sunny, CNRExt.overcast, PKLot.val], test=[CNRExt.sunny, CNRExt.overcast, PKLot.test]),

    # # Inter-camera experiments
    # pyffe.Experiment(model, solver, CNRExt.camera8, 
    #     val=[CNRExt.camera1, 
    #          CNRExt.camera2,
    #          CNRExt.camera3,
    #          CNRExt.camera4,
    #          CNRExt.camera5,
    #          CNRExt.camera6,
    #          CNRExt.camera7,
    #          PKLot.val,
    #          CNRExt.camera9],
    #     test=[CNRExt.camera1, 
    #          CNRExt.camera2,
    #          CNRExt.camera3,
    #          CNRExt.camera4,
    #          CNRExt.camera5,
    #          CNRExt.camera6,
    #          CNRExt.camera7,
    #          PKLot.test,
    #          CNRExt.camera9]),
             
    # pyffe.Experiment(model, solver, CNRExt.camera1,
    #     val=[PKLot.val,
    #          CNRExt.camera2, 
    #          CNRExt.camera3,
    #          CNRExt.camera4,
    #          CNRExt.camera5,
    #          CNRExt.camera6,
    #          CNRExt.camera7,
    #          CNRExt.camera8,
    #          CNRExt.camera9],
    #     test=[PKLot.test,
    #          CNRExt.camera2, 
    #          CNRExt.camera3,
    #          CNRExt.camera4,
    #          CNRExt.camera5,
    #          CNRExt.camera6,
    #          CNRExt.camera7,
    #          CNRExt.camera8,
    #          CNRExt.camera9]),
   
    # # Custom training 
    # pyffe.Experiment(bigmodel, solver, Custom.test, val=[Custom.test], test=[Custom.test]),
    
    # ## Pretrained test
    # # - AlexNet-on-CNRParkAB_all-val-CNRPark-EXT_val - 53.59%
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[CNRExt.val], test=[Custom.test]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.685083
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom.test]),
    
    # # - AlexNet-on-Combined_CNRParkAB_Ext_train_C1C8-val-CNRPark-EXT_val - 0.497238
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[CNRExt.val], test=[Custom.test]),
    
    # # - AlexNet-on-Combined_CNRParkAB_Ext_train_C1C8-val-PKLot_val - 0.640884
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train_C1C8, val=[PKLot.val], test=[Custom.test]),
    
    # # - AlexNet-on-Combined_CNRParkAB_Ext_train-val-CNRPark-EXT_val - 0.497238
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train, val=[CNRExt.val], test=[Custom.test]),
    
    # # - AlexNet-on-Combined_CNRParkAB_Ext_train-val-PKLot_val - 0.632597
    # pyffe.Experiment(bigmodel, solver, Combined.CNRParkAB_Ext_train, val=[PKLot.val], test=[Custom.test]),
    
    # # - AlexNet-on-PKLot_train-val-CNRPark-EXT_val - 0.497238
    # pyffe.Experiment(bigmodel, solver, PKLot.train, val=[CNRExt.val], test=[Custom.test]),
    
    # # - AlexNet-on-PKLot_train-val-PKLot_val - 0.649171
    # pyffe.Experiment(bigmodel, solver, PKLot.train, val=[PKLot.val], test=[Custom.test]),
    
    
    
    # --------------------------------- Custom ------------------------------------ #
    # # - AlexNet-on-Custom_all-val-Custom_all ~ 55%
    # pyffe.Experiment(bigmodel, solver, Custom.all, val=[Custom.all], test=[CNRParkAB.all]), 
    
    # # - AlexNet-on-Custom_train-val-Custom_val-ts-Custom_test - 0.885714
    # pyffe.Experiment(bigmodel, solver, Custom.train, val=[Custom.val], test=[Custom.test]), 
    
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.377049
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom.empty]), 
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 1
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom.occupied]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.576471
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom.real_empty]),
    
    
    # --------------------------------- Custom_GIST_220712 ------------------------------------ #
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.802326
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_GIST_220712.empty]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 1
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_GIST_220712.occupied]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.906593
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_GIST_220712.all]),
    
    
    
    # --------------------------------- Custom_Bluecom_220718 ------------------------------------ #
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220718.empty]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220718.occupied]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220718.all]),
    
    
    # --------------------------------- Custom_Bluecom_220801 ------------------------------------ #
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.684615
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220801.empty]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 1
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220801.occupied]),
    
    # # - AlexNet-on-CNRParkAB_all-val-PKLot_val - 0.831967
    # pyffe.Experiment(bigmodel, solver, CNRParkAB.all, val=[PKLot.val], test=[Custom_Bluecom_220801.all]),
    
    
    
    
    # # --------------------------------- Cross validation ------------------------------------ #
    # ## Custom_all
    # # - AlexNet-on-Custom_all-val-Custom_all
    # # - ts-Custom_all - 0.991713
    # # - ts-Custom_occupied - 1
    # # - ts-Custom_empty - 0.983607
    # # - ts-Custom_real_empty - 1
    # # - ts-Custom_all_with_real_empty - 1
    # # - ts-Custom_GIST_220712_all - 1
    # # - ts-Custom_GIST_220712_occupied - 1
    # # - ts-Custom_GIST_220712_empty - 1
    # # - ts-Custom_Bluecom_220801_all - 0.938525
    # # - ts-Custom_Bluecom_220801_occupied - 1
    # # - ts-Custom_Bluecom_220801_empty - 0.884615
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom.all, 
    #                  val=[Custom.all], 
    #                  test=[Custom.all, Custom.occupied, Custom.empty, Custom.real_empty, Custom.all_with_real_empty, Custom_GIST_220712.all, Custom_GIST_220712.occupied, Custom_GIST_220712.empty, Custom_Bluecom_220801.all, Custom_Bluecom_220801.occupied, Custom_Bluecom_220801.empty),
    
    # ## Custom_all_with_real_empty
    # # - AlexNet-on-Custom_all_with_real_empty-val-Custom_all_with_real_empty
    # # - ts-Custom_all - 0.994475
    # # - ts-Custom_occupied - 1
    # # - ts-Custom_empty - 0.989071
    # # - ts-Custom_real_empty - 1
    # # - ts-Custom_all_with_real_empty - 1
    # # - ts-Custom_GIST_220712_all - 0.978022
    # # - ts-Custom_GIST_220712_occupied - 0.958333
    # # - ts-Custom_GIST_220712_empty - 1
    # # - ts-Custom_Bluecom_220801_all - 0.967213
    # # - ts-Custom_Bluecom_220801_occupied - 0.938596
    # # - ts-Custom_Bluecom_220801_empty - 0.992308
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom.all_with_real_empty, 
    #                  val=[Custom.all_with_real_empty], 
    #                  test=[Custom.all, Custom.occupied, Custom.empty, Custom.real_empty, Custom.all_with_real_empty, Custom_GIST_220712.all, Custom_GIST_220712.occupied, Custom_GIST_220712.empty, Custom_Bluecom_220801.all, Custom_Bluecom_220801.occupied, Custom_Bluecom_220801.empty]),
    
    # ## Custom_GIST_220712
    # # - AlexNet-on-Custom_GIST_220712_all-val-Custom_GIST_220712_all
    # # - ts-Custom_all - 0.646409
    # # - ts-Custom_occupied - 0.290503
    # # - ts-Custom_empty - 0.994536
    # # - ts-Custom_real_empty - 1
    # # - ts-Custom_all_with_real_empty - 0.518939
    # # - ts-Custom_GIST_220712_all - 0.961538
    # # - ts-Custom_GIST_220712_occupied - 0.927083
    # # - ts-Custom_GIST_220712_empty - 1
    # # - ts-Custom_Bluecom_220801_all - 0.868852
    # # - ts-Custom_Bluecom_220801_occupied - 0.745614
    # # - ts-Custom_Bluecom_220801_empty - 0.976923
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_GIST_220712.all, 
    #                  val=[Custom_GIST_220712.all], 
    #                  test=[Custom.all, Custom.occupied, Custom.empty, Custom.real_empty, Custom.all_with_real_empty, Custom_GIST_220712.all, Custom_GIST_220712.occupied, Custom_GIST_220712.empty, Custom_Bluecom_220801.all, Custom_Bluecom_220801.occupied, Custom_Bluecom_220801.empty]),
    
    # ## Custom_Bluecom_220801
    # # - AlexNet-on-Custom_Bluecom_220801_all-val-Custom_Bluecom_220801_all
    # # - ts-Custom_all - 0.828729
    # # - ts-Custom_occupied - 1
    # # - ts-Custom_empty - 0.661202
    # # - ts-Custom_real_empty - 0.882353
    # # - ts-Custom_all_with_real_empty - 0.962121
    # # - ts-Custom_GIST_220712_all - 0.994505
    # # - ts-Custom_GIST_220712_occupied - 1
    # # - ts-Custom_GIST_220712_empty - 0.988372
    # # - ts-Custom_Bluecom_220801_all - 0.991803
    # # - ts-Custom_Bluecom_220801_occupied - 1
    # # - ts-Custom_Bluecom_220801_empty - 0.984615
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220801.all, 
    #                  val=[Custom_Bluecom_220801.all], 
    #                  test=[Custom.all, Custom.occupied, Custom.empty, Custom.real_empty, Custom.all_with_real_empty, Custom_GIST_220712.all, Custom_GIST_220712.occupied, Custom_GIST_220712.empty, Custom_Bluecom_220801.all, Custom_Bluecom_220801.occupied, Custom_Bluecom_220801.empty]),
        
    
    # ## Custom_Bluecom_220801
    # # - AlexNet-on-Custom_Bluecom_220801_all-val-Custom_Bluecom_220801_all
    # # Custom_Bluecom_220802.empty_144880_009    - 0.7
    # # Custom_Bluecom_220802.occupied_144880_009 - 1
    # # Custom_Bluecom_220802.empty_150606_010    - 0.287037
    # # Custom_Bluecom_220802.occupied_150606_010 - 1
    # # Custom_Bluecom_220802.empty_162722_011    - 0.697674
    # # Custom_Bluecom_220802.occupied_162722_011 - 1
    # # Custom_Bluecom_220802.empty_164119_015    - 0.454545
    # # Custom_Bluecom_220802.occupied_164119_015 - 1
    # # Custom_Bluecom_220802.empty_164625_016    - 0.333333
    # # Custom_Bluecom_220802.occupied_164625_016 - 1
    # # Custom_Bluecom_220802.empty_164941_017    - 0.117647
    # # Custom_Bluecom_220802.occupied_164941_017 - 1
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220801.all, 
    #                  val=[Custom_Bluecom_220801.all], 
    #                  test=[Custom_Bluecom_220802.empty_144880_009, Custom_Bluecom_220802.occupied_144880_009,
    #                        Custom_Bluecom_220802.empty_150606_010, 
    #                        Custom_Bluecom_220802.occupied_150606_010,
    #                        Custom_Bluecom_220802.empty_162722_011, 
    #                        Custom_Bluecom_220802.occupied_162722_011,
    #                        Custom_Bluecom_220802.empty_164119_015, 
    #                        Custom_Bluecom_220802.occupied_164119_015,
    #                        Custom_Bluecom_220802.empty_164625_016, 
    #                        Custom_Bluecom_220802.occupied_164625_016,
    #                        Custom_Bluecom_220802.empty_164941_017, 
    #                        Custom_Bluecom_220802.occupied_164941_017]),
    
    # ## Custom_Bluecom_Total
    # # - AlexNet-on-Custom_Bluecom_Total_all-val-Custom_Bluecom_Total_all
    # # Custom_Bluecom_Total_all - 0.984975
    # # Custom_Bluecom_Total_occupied - 0.997245
    # # Custom_Bluecom_Total_empty - 0.979641
    # # Custom_all_with_real_empty - 0.992424
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_Total.all, 
    #                  val=[Custom_Bluecom_Total.all], 
    #                  test=[Custom_Bluecom_Total.all, Custom_Bluecom_Total.occupied, Custom_Bluecom_Total.empty, Custom.all_with_real_empty]),
    
    ## Custom_Bluecom_220801_empty_extension
    # - AlexNet-on-Custom_Bluecom_220801_all_with_empty_extension-val-Custom_Bluecom_220801_all_with_empty_extension
    # Custom_Bluecom_220801_occupied            - 1
    # Custom_Bluecom_220801_empty               - 0.992308
    # Custom_Bluecom_220801_empty_extension     - 0.848168
    # Custom_real_empty                         - 0.682353
    # Custom_occupied                           - 0.98324
    # Custom_GIST_220712_empty                  - 0.965116
    # Custom_GIST_220712_occupied               - 1
    # Custom_Bluecom_220802.empty_144880_009    - 0.94
    # Custom_Bluecom_220802.occupied_144880_009 - 1
    # Custom_Bluecom_220802.empty_150606_010    - 0.378505
    # Custom_Bluecom_220802.occupied_150606_010 - 0.987952
    # Custom_Bluecom_220802.empty_162722_011    - 0.581395
    # Custom_Bluecom_220802.occupied_162722_011 - 1
    # Custom_Bluecom_220802.empty_164119_015    - 0.636364
    # Custom_Bluecom_220802.occupied_164119_015 - 0.985294
    # Custom_Bluecom_220802.empty_164625_016    - 0.833333
    # Custom_Bluecom_220802.occupied_164625_016 - 1
    # Custom_Bluecom_220802.empty_164941_017    - 0.941176
    # Custom_Bluecom_220802.occupied_164941_017 - 0.972222
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220801.all_with_empty_extension, 
    #                  val=[Custom_Bluecom_220801.all_with_empty_extension], 
    #                  test=[Custom_Bluecom_220801.occupied, 
    #                        Custom_Bluecom_220801.empty, 
    #                        Custom_Bluecom_220801.empty_extension, 
    #                        Custom.empty,
    #                        Custom.real_empty,
    #                        Custom.occupied,
    #                        Custom_GIST_220712.empty,
    #                        Custom_GIST_220712.occupied,
    #                        Custom_Bluecom_220802.empty_144880_009, 
    #                        Custom_Bluecom_220802.occupied_144880_009,
    #                        Custom_Bluecom_220802.empty_150606_010, 
    #                        Custom_Bluecom_220802.occupied_150606_010,
    #                        Custom_Bluecom_220802.empty_162722_011, 
    #                        Custom_Bluecom_220802.occupied_162722_011,
    #                        Custom_Bluecom_220802.empty_164119_015, 
    #                        Custom_Bluecom_220802.occupied_164119_015,
    #                        Custom_Bluecom_220802.empty_164625_016, 
    #                        Custom_Bluecom_220802.occupied_164625_016,
    #                        Custom_Bluecom_220802.empty_164941_017, 
    #                        Custom_Bluecom_220802.occupied_164941_017]),
    
    ## Custom_Bluecom_220801_empty_extension
    # - AlexNet-on-Custom_Bluecom_220801_all_with_empty_extension-val-Custom_Bluecom_220802_all
    # Custom_Bluecom_220801_occupied            - 
    # Custom_Bluecom_220801_empty               - 
    # Custom_Bluecom_220801_empty_extension     - 
    # Custom_real_empty                         - 
    # Custom_occupied                           - 
    # Custom_GIST_220712_empty                  - 
    # Custom_GIST_220712_occupied               - 
    # Custom_Bluecom_220802.empty_144880_009    - 
    # Custom_Bluecom_220802.occupied_144880_009 - 
    # Custom_Bluecom_220802.empty_150606_010    - 
    # Custom_Bluecom_220802.occupied_150606_010 - 
    # Custom_Bluecom_220802.empty_162722_011    - 
    # Custom_Bluecom_220802.occupied_162722_011 - 
    # Custom_Bluecom_220802.empty_164119_015    - 
    # Custom_Bluecom_220802.occupied_164119_015 - 
    # Custom_Bluecom_220802.empty_164625_016    - 
    # Custom_Bluecom_220802.occupied_164625_016 - 
    # Custom_Bluecom_220802.empty_164941_017    - 
    # Custom_Bluecom_220802.occupied_164941_017 - 
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220801.all_with_empty_extension, 
    #                  val=[Custom_Bluecom_220802.all], 
    #                  test=[Custom_Bluecom_220801.occupied, 
    #                        Custom_Bluecom_220801.empty, 
    #                        Custom_Bluecom_220801.empty_extension, 
    #                        Custom.real_empty,
    #                        Custom.occupied,
    #                        Custom_GIST_220712.empty,
    #                        Custom_GIST_220712.occupied,
    #                        Custom_Bluecom_220802.empty_144880_009, 
    #                        Custom_Bluecom_220802.occupied_144880_009,
    #                        Custom_Bluecom_220802.empty_150606_010, 
    #                        Custom_Bluecom_220802.occupied_150606_010,
    #                        Custom_Bluecom_220802.empty_162722_011, 
    #                        Custom_Bluecom_220802.occupied_162722_011,
    #                        Custom_Bluecom_220802.empty_164119_015, 
    #                        Custom_Bluecom_220802.occupied_164119_015,
    #                        Custom_Bluecom_220802.empty_164625_016, 
    #                        Custom_Bluecom_220802.occupied_164625_016,
    #                        Custom_Bluecom_220802.empty_164941_017, 
    #                        Custom_Bluecom_220802.occupied_164941_017]),
    
    
    ## Custom_Bluecom_220802
    # - AlexNet-on-Custom_Bluecom_220802_all-val-Custom_Bluecom_220802_all
    # Custom_Bluecom_220801_occupied            - 0.973684
    # Custom_Bluecom_220801_empty               - 1
    # Custom_Bluecom_220801_empty_extension     - 0.926702
    # Custom_real_empty                         - 1
    # Custom_occupied                           - 1
    # Custom_GIST_220712_empty                  - 1
    # Custom_GIST_220712_occupied               - 0.947917
    # Custom_Bluecom_220802.all                 - 0.986971
    # Custom_Bluecom_220802.empty               - 0.978082
    # Custom_Bluecom_220802.occupied            - 1
    # Custom_Bluecom_220802.empty_144880_009    - 1
    # Custom_Bluecom_220802.empty_150606_010    - 0.985981
    # Custom_Bluecom_220802.empty_162722_011    - 0.930233
    # Custom_Bluecom_220802.empty_164119_015    - 0.818182
    # Custom_Bluecom_220802.empty_164625_016    - 1
    # Custom_Bluecom_220802.empty_164941_017    - 1
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220802.all,
    #                  val=[Custom_Bluecom_220802.all], 
    #                  test=[Custom_Bluecom_220801.occupied, 
    #                        Custom_Bluecom_220801.empty, 
    #                        Custom_Bluecom_220801.empty_extension, 
    #                        Custom.empty,
    #                        Custom.real_empty,
    #                        Custom.occupied,
    #                        Custom_GIST_220712.empty,
    #                        Custom_GIST_220712.occupied,
    #                        Custom_Bluecom_220802.all,
    #                        Custom_Bluecom_220802.empty,
    #                        Custom_Bluecom_220802.occupied,
    #                        Custom_Bluecom_220802.empty_144880_009, 
    #                        Custom_Bluecom_220802.empty_150606_010, 
    #                        Custom_Bluecom_220802.empty_162722_011, 
    #                        Custom_Bluecom_220802.empty_164119_015, 
    #                        Custom_Bluecom_220802.empty_164625_016, 
    #                        Custom_Bluecom_220802.empty_164941_017]),
    
    ## Custom_Bluecom_220802
    # Custom_141753_001.all         - 0.93
    # Custom_141753_001.empty       - 0.852941
    # Custom_141753_001.occupied    - 0.969697
    # pyffe.Experiment(bigmodel, solver, 
    #                  Custom_Bluecom_220802.all,
    #                  val=[Custom_Bluecom_220802.all], 
    #                  test=[Custom_141753_001.all,
    #                        Custom_141753_001.empty,
    #                        Custom_141753_001.occupied]),
    
    
    ## Custom_Bluecom_220802
    # Custom_dji_dji_0001_001-002_00601 -
    # Custom_dji_dji_0001_002-014 -
    # Custom_dji_dji_0002_001-015 -
    # Custom_dji_dji_0002_002 -
    # Custom_dji_dji_0003_001-007 -
    # Custom_dji_dji_0003_002 -
    # Custom_dji_dji_0004_001-008 -
    # Custom_dji_dji_0004_002-009 -
    # Custom_dji_dji_0004_003 -
    # Custom_dji_dji_0006_001-011 -
    # Custom_dji_dji_0006_002-012 -
    # Custom_dji_dji_0006_003 -
    # Custom_dji_dji_0007_001-005 -
    # Custom_dji_dji_0007_002 -
    # Custom_dji_dji_0009_001 -
    # Custom_dji_dji_0011_002 -
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Bluecom_220802.all,
    #                 val=[Custom_Bluecom_220802.all], 
    #                 test=[  Custom_dji.dji_0001_001_002_00601_empty,
    #                         Custom_dji.dji_0001_001_002_00601_occupied,
    #                         Custom_dji.dji_0001_002_014_empty,
    #                         Custom_dji.dji_0001_002_014_occupied,
    #                         Custom_dji.dji_0002_001_015_empty,
    #                         Custom_dji.dji_0002_001_015_occupied,
    #                         Custom_dji.dji_0002_002_empty,
    #                         Custom_dji.dji_0002_002_occupied,
    #                         Custom_dji.dji_0003_001_007_empty,
    #                         Custom_dji.dji_0003_001_007_occupied,
    #                         Custom_dji.dji_0003_002_empty,
    #                         Custom_dji.dji_0003_002_occupied,
    #                         Custom_dji.dji_0004_001_008_empty,
    #                         Custom_dji.dji_0004_001_008_occupied,
    #                         Custom_dji.dji_0004_002_009_empty,
    #                         Custom_dji.dji_0004_002_009_occupied,
    #                         Custom_dji.dji_0004_003_empty,
    #                         Custom_dji.dji_0004_003_occupied,
    #                         Custom_dji.dji_0006_001_011_empty,
    #                         Custom_dji.dji_0006_001_011_occupied,
    #                         Custom_dji.dji_0006_002_012_empty,
    #                         Custom_dji.dji_0006_002_012_occupied,
    #                         Custom_dji.dji_0006_003_empty,
    #                         Custom_dji.dji_0006_003_occupied,
    #                         Custom_dji.dji_0007_001_005_empty,
    #                         Custom_dji.dji_0007_001_005_occupied,
    #                         Custom_dji.dji_0007_002_empty,
    #                         Custom_dji.dji_0007_002_occupied,
    #                         Custom_dji.dji_0009_001_empty,
    #                         Custom_dji.dji_0009_001_occupied,
    #                         Custom_dji.dji_0011_002_empty,
    #                         Custom_dji.dji_0011_002_occupied]),
    
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Bluecom_220802.all,
    #                 val=[Custom_Bluecom_220802.all],
    #                 test=[Custom_insta.all_163750_014, Custom_insta.empty_163750_014, Custom_insta.occupied_163750_014]),
    
    
    # # Custom_Paper
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.train_all,
    #                 val=[Custom_Paper.train_all], 
    #                 test=[Custom_Paper.train_all, Custom_Paper.train_empty, Custom_Paper.train_occupied,
    #                       Custom_Paper.test_all, Custom_Paper.test_empty, Custom_Paper.test_occupied]),
    
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.test_all,
    #                 val=[Custom_Paper.test_all], 
    #                 test=[Custom_Paper.train_all, Custom_Paper.train_empty, Custom_Paper.train_occupied,
    #                       Custom_Paper.test_all, Custom_Paper.test_empty, Custom_Paper.test_occupied]),
    
    # Custom_Paper_5_fold
    # 1234-5
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.fold1234_all,
    #                 val=[Custom_Paper.fold1234_all], 
    #                 test=[Custom_Paper.fold5_all, Custom_Paper.fold5_empty, Custom_Paper.fold5_occupied]),
    
    # # 1235-4
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.fold1235_all,
    #                 val=[Custom_Paper.fold1235_all], 
    #                 test=[Custom_Paper.fold4_all, Custom_Paper.fold4_empty, Custom_Paper.fold4_occupied]),
    
    # #1245-3
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.fold1245_all,
    #                 val=[Custom_Paper.fold1245_all], 
    #                 test=[Custom_Paper.fold3_all, Custom_Paper.fold3_empty, Custom_Paper.fold3_occupied]),
    
    #1345-2
    pyffe.Experiment(bigmodel, solver, 
                    Custom_Paper.fold1345_all,
                    val=[Custom_Paper.fold1345_all], 
                    test=[Custom_Paper.fold2_all, Custom_Paper.fold2_empty, Custom_Paper.fold2_occupied]),
    
    # #2345-1
    # pyffe.Experiment(bigmodel, solver, 
    #                 Custom_Paper.fold2345_all,
    #                 val=[Custom_Paper.fold2345_all], 
    #                 test=[Custom_Paper.fold1_all, Custom_Paper.fold1_empty, Custom_Paper.fold1_occupied]),
        
    
]

for exp in exps:
    exp.setup('runs/')
    exp.run(plot=False, resume=False) # run without live plot
    exp.run_test()
    
    #print(exp.get_features(exp.extract_features(dataset=Custom_Bluecom_220718.all, snapshot_iter=1020)))
    #print(exp.get_accuracy_at(1020, Custom_Bluecom_220718.all))
    #exp.my_test(10764, Custom_Paper.test_empty)
    #exp.my_test(10764, Custom_Paper.test_occupied)
    #exp.my_test(10764, Custom_Paper.train_occupied)


#pyffe.summarize(exps).to_csv('results.csv')


