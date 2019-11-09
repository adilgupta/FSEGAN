import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from data_loader import Dataset
from networks import G, D, dl_model
from torch import optim
from operator import add
import pdb


def get_phones_td(sig, rate, dl_model):
    feat, energy = fbank(sig, samplerate=rate, nfilt=38, winfunc=np.hamming)
    #feat = np.log(feat)
    tsteps, hidden_dim = feat.shape
    feat_log_full = np.reshape(np.log(feat), (1, tsteps, hidden_dim))
    lens = np.array([tsteps])
    inputs, lens = torch.from_numpy(np.array(feat_log_full)).float(), torch.from_numpy(np.array(lens)).long()
    id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            lens = lens.cuda()
        # Pass through model
        outputs = dl_model.model(inputs, lens).cpu().numpy()
        # Since only one example per batch and ignore blank token
        outputs = outputs[0, :, :-1]
        softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

    outputs, mapping = softmax, id_to_phone
    final_lattice = generate_lattice(outputs, 0.2)
    #db.set_trace()
    #print(np.argmax(outputs, axis=1))
    phones = [[mapping[x[0]] for x in l] for l in final_lattice]
    return np.argmax(outputs, axis = 1)# phones

def get_phones_feat_map(feat, dl_model, lens):
    #tsteps = feat.shape[2]
    #feat_log_full = feat
    #lens = np.array([tsteps])
    inputs = feat #torch.from_numpy(np.array(feat_log_full)).float()
    id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}
    #with torch.no_grad():
        #if torch.cuda.is_available():
    inputs = inputs.cuda()
    lens = lens.cuda()
    # Pass through model

    outputs = dl_model.model(inputs, lens)#.cpu().numpy()
    # Since only one example per batch and ignore blank token
    outputs = outputs[:, :, :-1]
    m = nn.Softmax(dim = 2)
    softmax = m(outputs)
    #softmax = torch.exp(outputs) / torch.sum(torch.exp(outputs), dim=2)[:, None]

    #outputs, mapping = softmax, id_to_phone
    #final_lattice = generate_lattice(outputs, 0.2)
    #print(np.argmax(outputs, axis=1))

    #phones = [[mapping[x[0]] for x in l] for l in final_lattice]
    return softmax#phones


def generate_lattice(outputs, h_spike):
    tsteps, num_phones = outputs.shape
    lattice = [[] for i in range(tsteps)]
    for i in range(tsteps):
        for j in range(num_phones):
            if outputs[i][j] >= h_spike:
                lattice[i].append((j, outputs[i][j]))

    # Collapse consecutive
    final_lattice = []
    previous_phones = [x[0] for x in lattice[0]]
    prev_sum = [x[1] for x in lattice[0]]
    num = 1

    for l in lattice[1:]:
        ids, vals = [x[0] for x in l], [x[1] for x in l]

        if ids == previous_phones:
            num += 1
            prev_sum = list(map(add, prev_sum, vals))
        else:
            final_lattice.append(list(zip(previous_phones, [x / num for x in prev_sum])))
            previous_phones = ids
            prev_sum = vals
            num = 1

    final_lattice.append(list(zip(previous_phones, [x / num for x in prev_sum])))

    return final_lattice

def phn_loss(input_, target_):
    loss = nn.CrossEntropyLoss()
    losses = 0
    for i in range(input_.shape[0]):
        losses += loss(input_[i,:,:], torch.argmax(target_[i,:,:], dim = 1))
    return losses/(i+1)

if __name__ == "__main__":
    train_dataset = Dataset()
    batch_size = 25
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    gen = G()
    dis = D()
    if torch.cuda.is_available():
        dis = nn.DataParallel(dis.cuda())
        gen = nn.DataParallel(gen.cuda())
    print("# generator parameters:", sum(param.numel() for param in gen.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in dis.parameters()))
    g_optimizer = optim.Adam(gen.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(dis.parameters(), lr=0.0001)

    cond_loss_arr = []
    dis_loss_arr = []
    gen_loss_arr = []
    g_loss_arr = []
    d_loss_arr = []

    mean_gen_loss = []
    mean_dis_loss = []
    mean_phn_loss = []

    dl_model = dl_model("test_one")
    dl_model.model.train()

    #phn_loss = nn.CrossEntropyLoss()
    for epoch in range(1000):
        train_bar = tqdm(train_data_loader)
        phn_loss_arr = []
        g_loss_arr = []
        for train_noisy_, train_clean_ in train_bar:
            if torch.cuda.is_available():
                train_clean, train_noisy = train_clean_.cuda(), train_noisy_.cuda()
                train_clean, train_noisy = Variable(train_clean), Variable(train_noisy)
#----------------------------------------train discriminator----------------------------------------
                """
                dis.zero_grad()
                gen.zero_grad()
                d_out_clean = dis(train_clean)
                clean_loss = torch.mean((d_out_clean - 1.0)**2)
                enhan_outputs = gen(train_noisy)
                d_out_enhan = dis(enhan_outputs)
                noisy_loss = torch.mean(d_out_enhan**2)
                #d_loss = -torch.mean(torch.abs(outputs_gen - outputs_c))
                d_loss = -1*torch.mean(torch.abs(d_out_clean - d_out_enhan)) #clean_loss + noisy_loss
                d_loss.backward()
                d_optimizer.step()
                for p in dis.parameters():
                    p.data.clamp_(-0.05, 0.05)
                """
#----------------------------------------train generator-----------------------------------------------

                gen.zero_grad()
                dis.zero_grad()
                generated_outputs = gen(train_noisy)
                outputs = dis(generated_outputs)
                outputs_c = dis(train_clean)
                #g_loss_ = torch.mean(torch.abs(outputs_gen - outputs_c))
                g_loss_ = torch.mean(torch.abs(outputs-outputs_c))#0.5*torch.mean((outputs-1.0)**2)  #---------------------------adversarial loss

                g_cond_loss = 100*torch.mean(torch.abs((generated_outputs - train_clean)))#------------------------L1 loss
#------------------------------------------------------------phone loss---------------------------------------------------------------------------------
                tsteps = train_clean.shape[2]
                lens = np.ones(train_clean.shape[0])*tsteps
                inputs_enhanced, inputs_clean, lens = generated_outputs[:,0,:,:], train_clean[:,0,:,:], torch.from_numpy(np.array(lens)).long()
                dl_model.model.zero_grad()
                phones_enhanced = get_phones_feat_map(inputs_enhanced, dl_model, lens)
                phones_clean = get_phones_feat_map(inputs_clean, dl_model, lens)
                phone_loss = phn_loss(phones_enhanced, phones_clean)

                #pdb.set_trace()

                """
                #tsteps = feat.shape[2]
                feat_log_full = np.reshape(feat, (1, tsteps, hidden_dim))
                #lens = np.array([tsteps])
                inputs = torch.from_numpy(np.array(feat_log_full)).float()
                id_to_phone = {v[0]: k for k, v in dl_model.model.phone_to_id.items()}
                #with torch.no_grad():
                    #if torch.cuda.is_available():
                inputs = inputs.cuda()
                lens = lens.cuda()
                # Pass through model
                outputs = dl_model.model(inputs, lens).cpu().numpy()
                # Since only one example per batch and ignore blank token
                outputs = outputs[0, :, :-1]
                softmax = np.exp(outputs) / np.sum(np.exp(outputs), axis=1)[:, None]

                outputs, mapping = softmax, id_to_phone


                tsteps = train_clean.shape[2]
                dl_model_ = dl_model("test_one")
                #dl_model.eval()
                #dl_model_.model.zero_grad()
                id_to_phone = {v[0]: k for k, v in dl_model_.model.phone_to_id.items()}
                lens = np.ones(train_noisy.shape[0])*tsteps
                inputs_noisy, inputs_clean, lens = generated_outputs[:,0,:,:], train_clean[:,0,:,:], torch.from_numpy(np.array(lens)).long()
                #dl_model.model.train()
                outputs_noisy = dl_model_.model(inputs_noisy, lens)#.cpu().detach().numpy()
                outputs_clean = dl_model_.model(inputs_clean, lens)#.cpu().detach().numpy()
                # Since only one example per batch and ignore blank token
                outputs_noisy = outputs_noisy[0, :, :-1]
                outputs_clean = outputs_clean[0, :, :-1]
                softmax_clean = torch.exp(outputs_clean) / torch.sum(torch.exp(outputs_clean), dim=1)[:, None]
                softmax_noisy = torch.exp(outputs_noisy) / torch.sum(torch.exp(outputs_noisy), dim=1)[:, None]
                outputs_clean, mapping = softmax_clean, id_to_phone
                outputs_noisy, mapping = softmax_noisy, id_to_phone
                #final_lattice_clean = generate_lattice(outputs_clean, 0.2)
                #final_lattice_noisy = generate_lattice(outputs_noisy, 0.2)
                #print(final_lattice, ([len(x) for x in final_lattice if len(x) != 1]))
                #pdb.set_trace()

                #phones_clean = [[mapping[x[0]] for x in l] for l in final_lattice_clean]
                #phones_noisy = [[mapping[x[0]] for x in l] for l in final_lattice_noisy]

                #a = torch.argmax(outputs_noisy, dim = 1)
                a_ = torch.argmax(outputs_clean, dim = 1)
                phone_loss = phn_loss(outputs_noisy, a_)
                #pdb.set_trace()
                #phone_loss = phn_loss(torch.argmax(outputs_noisy, dim = 1), torch.argmax(outputs_clean, dim = 1))

                #print(phones_clean)
                #print(phones_noisy)
                """
#------------------------------------------------------------loss sum--------------------------------------------------------------
                """
                alp = 1
                bet = 1
                gam = 100
                g_loss = alp*g_cond_loss + bet*g_loss_ + gam*phone_loss
                """

                g_loss = g_cond_loss*0 + g_loss_*0 + phone_loss
                g_loss.backward()
                g_optimizer.step()
                train_bar.set_description('Epoch {}: loss {:.4f}'.format(epoch+1, phone_loss.item()))
                g_loss_arr.append(g_loss.detach().cpu().item())
                #d_loss_arr.append(d_loss.detach().cpu().item())
                phn_loss_arr.append(phone_loss.detach().cpu().item())

        print( np.mean(phn_loss_arr))
        mean_gen_loss.append(np.mean(g_loss_arr))
        #mean_dis_loss.append(np.mean(d_loss_arr))
        mean_phn_loss.append(np.mean(phn_loss_arr))
                #train_bar.set_description('Epoch {}: loss_gen {:.4f}, loss_dis {:.4f}, g_loss_ {:.4f}, g_cond_loss {:.4f}, phone_loss {:.4f}'.format(epoch+1, g_loss.item(), d_loss.item(), g_loss_.item(), g_cond_loss.item(), phone_loss.item()))

#-----------------------------------------------------------saving model------------------------------------------------------------

        torch.save(gen.state_dict(), '/media/Sharedata/adil/timit/epochs_adv_phn_2/gen_epoch_%d.pth' %(epoch))
        #torch.save(dis.state_dict(), '/media/Sharedata/adil/timit/epochs_adv_phn_2/dis_epoch_%d.pth' %(epoch))
        #np.savetxt('/media/Sharedata/adil/timit/losses_adv_phn_2/mean_dis_loss.txt', mean_dis_loss)
        np.savetxt('/media/Sharedata/adil/timit/losses_adv_phn_2/mean_gen_loss.txt', mean_gen_loss)
        #phn_loss_arr.append(phone_loss)
        #cond_loss_arr.append(g_cond_loss)
        #dis_loss_arr.append(d_loss)
        #gen_loss_arr.append(g_loss)
        #np.savetxt('/media/Sharedata/adil/timit/losses/phone_loss.txt', phn_loss_arr)
        #np.savetxt('/media/Sharedata/adil/timit/losses/cond_loss.txt', cond_loss_arr)
        #np.savetxt('/media/Sharedata/adil/timit/losses/dis_loss.txt', dis_loss_arr)
        #np.savetxt('/media/Sharedata/adil/timit/losses/gen_loss.txt', gen_loss_arr)
