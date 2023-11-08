import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/SVDSAM/")

from data_utils import *
from model import *
from utils import *

label_names = ['Liver', 'Kidney', 'Pancreas', 'Vessels', 'Adrenals', 'Gall Bladder', 'Bones', 'Spleen']
# visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
# visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        # visualize_dict[ln] = visualize_li[i]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='config_tmp.yml',
                        help='data folder file path')

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')

    parser.add_argument('--gt_path', default='',
                        help='ground truth path')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    parser.add_argument('--labels_of_interest', default='Left Prograsp Forceps,Maryland Bipolar Forceps,Right Prograsp Forceps,Left Large Needle Driver,Right Large Needle Driver', help='labels of interest')

    parser.add_argument('--codes', default='1,2,1,3,3', help='numeric label to save per instrument')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    labels_of_interest = args.labels_of_interest.split(',')
    codes = args.codes.split(',')
    codes = [int(c) for c in codes]

    label_dict = {
            'Liver': [[100,0,100]],
            'Kidney': [[255,255,0]],
            'Pancreas': [[0,0,255]],
            'Vessels': [[255,0,0]],
            'Adrenals': [[0,255,255]],
            'Gall Bladder': [[0,255,0]],
            'Bones': [[255,255,255]],
            'Spleen': [[255,0,255]]
        }


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    if args.gt_path:
        os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='svdtuning')
    # model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device, training_strategy='lora')

    #legacy model support
    if args.pretrained_path:
        sdict = torch.load(args.pretrained_path, map_location=args.device)
        # for key in list(sdict.keys()):
        #     if 'sam_encoder.neck' in key:
        #         if '0' in key:
        #             new_key = key.replace('0','conv1')
        #         if '1' in key:
        #             new_key = key.replace('1','ln1')
        #         if '2' in key:
        #             new_key = key.replace('2','conv2')
        #         if '3' in key:
        #             new_key = key.replace('3','ln2')
        #         sdict[new_key] = sdict[key]
        #         _ = sdict.pop(key)
        #     if 'mask_decoder' in key:
        #         if 'trainable' in key:
        #             _ = sdict.pop(key)   
        
        model.load_state_dict(sdict,strict=True)
    model = model.to(args.device)
    model = model.eval()

    #load data transform
    data_transform = Ultrasound_Transform(config=data_config)

    #dice
    dices = []
    ious=[]

    #load data
    for i,img_name in enumerate(sorted(os.listdir(args.data_folder))):
        # if i%5!=0:
        #     continue
        img_path = (os.path.join(args.data_folder,img_name))
        if args.gt_path:
            gt_path = (os.path.join(args.gt_path,img_name))
            if not os.path.exists(gt_path):
                gt_path = (os.path.join(args.gt_path,img_name[:-4]+'.png'))
                if not os.path.exists(gt_path):
                    continue

        # print(img_path)
        img = torch.as_tensor(np.array(Image.open(img_path).convert("RGB")))
        img = img.permute(2,0,1)
        C,H,W = img.shape
        #make a dummy mask of shape 1XHXW
        if args.gt_path:
            label = np.array(Image.open(gt_path).convert("RGB"))
            temp = np.zeros((H,W)).astype('uint8')
            selected_color_list = label_dict[args.labels_of_interest]
            for c in selected_color_list:
                temp = temp | (np.all(np.where(label==c,1,0),axis=2))

            # plt.imshow(gold)
            # plt.show()
            mask = torch.Tensor(temp).unsqueeze(0)

        else:
            mask = torch.zeros((1,H,W))
        img, mask = data_transform(img, mask, is_train=False, apply_norm=True)
        mask = (mask>=0.5)+0

        #get positive point prompts
        _,y,x = torch.where(mask==1)
        pos_prompts = torch.cat([x.unsqueeze(1),y.unsqueeze(1)],dim=1)

        #get negative point prompts
        _,y_neg,x_neg = torch.where(mask==0)
        neg_prompts = (torch.cat([x_neg.unsqueeze(1),y_neg.unsqueeze(1)],dim=1))

        if len(y)>0:
            pos_point_idx = random.randint(0,y.shape[0]-1)
            neg_point_idx = random.randint(0,y_neg.shape[0]-1)
            # points = (torch.cat([pos_prompts[pos_point_idx].unsqueeze(0), neg_prompts[neg_point_idx].unsqueeze(0)],dim=0).unsqueeze(0).to(args.device), torch.Tensor([1,-1]).unsqueeze(0).to(args.device))
            points = (pos_prompts[pos_point_idx].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([1]).unsqueeze(0).to(args.device))
            # print(points[0].shape)
        else:
            neg_point_idx1 = random.randint(0,y_neg.shape[0]-1)
            neg_point_idx2 = random.randint(0,y_neg.shape[0]-1)
            # points = (torch.cat([neg_prompts[neg_point_idx1].unsqueeze(0), neg_prompts[neg_point_idx2].unsqueeze(0)],dim=0).unsqueeze(0).to(args.device), torch.Tensor([-1,-1]).unsqueeze(0).to(args.device))
            points = (neg_prompts[neg_point_idx1].unsqueeze(0).unsqueeze(0).to(args.device), torch.Tensor([-1]).unsqueeze(0).to(args.device))
            # print(points[0].shape)
            
        #get image embeddings
        img = img.unsqueeze(0).to(args.device)  #1XCXHXW
        img_embeds = model.get_image_embeddings(img)

        # generate masks for all labels of interest
        img_embeds_repeated = img_embeds.repeat(len(labels_of_interest),1,1,1)
        masks= model.get_masks_with_manual_prompts(img_embeds_repeated, points=points).cpu()
        
        # save_im = Image.fromarray(final_mask.numpy())
        # save_im.save(os.path.join(args.save_path,'preds', img_name))

        # plt.imshow(final_mask_rescaled,cmap='gray')
        # plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
        # plt.close()

        # print("label shape: ", label.shape)
        # plt.imshow(label[0], cmap='gray')
        # plt.show()

        if args.gt_path:
            plt.imshow((mask[0]), cmap='gray')
            plt.savefig(os.path.join(args.save_path,'rescaled_gt', img_name))
            plt.close()

        plt.imshow((masks[0]>=0.5), cmap='gray')
        if len(y)>0:
            plt.scatter(x[pos_point_idx], y[pos_point_idx], c='green')
            # plt.scatter(x_neg[neg_point_idx], y_neg[neg_point_idx], c='red')
        else:
            plt.scatter(x_neg[neg_point_idx1], y_neg[neg_point_idx1], c='red')
            # plt.scatter(x_neg[neg_point_idx2], y_neg[neg_point_idx2], c='red')
        plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
        plt.close()
        # 10/0

        

        # print("dice: ",dice_coef(label, (masks>0.5)+0))
        dices.append(dice_coef(mask, (masks>=0.5)+0))
        ious.append(iou_coef(mask, (masks>=0.5)+0))

    print(torch.mean(torch.Tensor(dices)))
    print(torch.mean(torch.Tensor(ious)))

if __name__ == '__main__':
    main()


        

