optimizer = torch.optim.Adam(  list(model3.parameters()) + list(net.parameters())  , lr=3e-4)

criterion = torch.nn.MSELoss().to(device)


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model3,net, device, data_len ,vector_w, mels,pitches, energies,  criterion, optimizer,iter_meter):
    full_loss=0
    # элементы у меня по убыванию там
    r = list(range(data_len))
    random.shuffle(r)
    for i in r:
        #audios , words, mels,durations = waveforms[0] ,vector_w, vector_m[0]  ,input_lengths
        word, mel,pitch, energy  = vector_w[i].to(torch.float32), mels[i].to(torch.float32) ,pitches[i].to(torch.float32), energies[i].to(torch.float32)  
       # print(audios , words, mels,durations)
        word, mel,pitch, energy = word.to(device),mel[None,:,:].to(device),pitch[None,:].to(device) , energy[None,:].to(device)

        optimizer.zero_grad()
      #  words = words[-1,-1,:]
        word = word[:,None]
      
        word = word.to(device)
       

        mel_info = net(mel)
       # mel_info.requires_grad_()

        mel_output, pitch_output,energy_output,length = model3(word, mel_info)
        #mel_output.requires_grad_()
      #  pitch_output.requires_grad_()
       # energy_output.requires_grad_()
       
        loss = criterion(mel_output,  mel[:,:,:length]) +  0.5*criterion(pitch_output,  pitch[:,:length] ) + 0.5*criterion(energy_output, energy[:,:length])
        loss.requires_grad = True
        # +  0.5*criterion(pitch_output,  pitch[:,:length] ) + 0.5*criterion(energy_output, energy[:,:length])
        
        loss.backward()
        #print(loss.grad)
      #  experiment.log_metric('loss', loss.item(), step=iter_meter.get())
       # experiment.log_metric('learning_rate', scheduler.get_lr(), step=iter_meter.get())

        optimizer.step()
         #   scheduler.step()
        iter_meter.step()
        #if i % 100 == 0 or i == data_len:
         #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i ,  loss.item()))
        print("I IS ",  i , "LOSS IS " , loss.item())
        full_loss +=loss.item() 
        
     #   out_audio = audio.cpu().float().numpy().astype(np.float32, order='C')
        
return(mel_output,full_loss)    




epochs = 10
iter_meter = IterMeter()
for epoch in range(1, epochs + 1):
      #trainn(model3, device,len(train_loader.dataset), waveforms ,vector_w, vector_m, criterion, optimizer,  iter_meter)
      last_mel , full_loss  =train(model3, net, device,100,text_padded, mel_pad, pitch_padded,energy_padded, criterion, optimizer,  iter_meter)
