class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 1, 3)
        self.fc1 = nn.Linear( 6624, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
       
    def forward(self, x):
       # print(x.size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

checkpoint_path = f"checkpoint_80000"
state_dict = {}
for k, v in torch.load(checkpoint_path , map_location=torch.device('cpu'))['state_dict'].items():
   state_dict[k[7:]]=v



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def get_mask_from_lengths(lengths):
    #print(lengths.device)
    max_len = torch.max(lengths).item()
    ids = lengths.new_tensor(torch.arange(0, max_len)) #torch.arange(0, max_len)       lengths.new_tensor(torch.arange(0, max_len)), giving some warning
    mask = (lengths.unsqueeze(1) <= ids).to(torch.bool)
    #print("mask", mask.device)
    return mask  
    
    



class MyFastSpeech(Model):
    def __init__(self):
        super(MyFastSpeech,self).__init__(hparams)
       # self.hp = hparams
        self.load_state_dict(state_dict)
        self.Embedding = nn.Linear(1, 256)                                               # NEED
      #  self.mylin2 =nn.Linear(256, 200) # to downsample output from encoders


        #self.mylin3 = nn.Linear(10, 256)  #to upsample output from mel-spectro 
        self.mylin3 = nn.Linear(10, 40)  #to upsample output from mel-spectro            # NEED


        #self.Embedding = nn.Embedding(hp.n_symbols, hp.symbols_embedding_dim)
        
        self.alpha1 = self.alpha1
        self.alpha2 = self.alpha2
        self.register_buffer= self.register_buffer
        self.dropout = self.dropout                                                      
        self.Encoder = self.Encoder
        self.Decoder =  self.Decoder 

        self.Duration =self.Duration                                                     # NEED
        self.Pitch = self.Pitch                                                          # NEED
        self.Energy =self.Energy                                                         # NEED
                                                                                         # but they are w the same architecture

        self.Projection =self.Projection                                                 # NEED


  
    def forward(self, text,melstext):
          

        #  text = text[:,:text_lengths.max().item()]                        #no need for this maybe [B, L]
         # melspec = melspec[:,:,:mel_lengths.max().item()]                #no need for this maybe [B, 80, T]
          ##alignments = alignments[:,:mel_lengths.max().item(),:text_lengths.max().item()]
          #print(text.device, durations.device, text_lengths.device, mel_lengths.device)
          mel,pitch,energy,length = self.inference(text.cuda(),melstext.cuda())
         
          return (mel,pitch,energy,length)

        
    def inference(self, text, melstext, alpha=1.0):
              
              #input - text_sequence (ndarray) (109,)
              
              device = torch.device("cuda" )
              ### Prepare Inference ###
              text_lengths = torch.tensor([text.shape[0]])       # [L]                              #torch.tensor([1, text.size(1)])
              text = text.unsqueeze(0) # .from_numpy(text)         #[1, L]

              ### Prepare Inputs ###
              encoder_input = self.Embedding(text).transpose(0,1)
              encoder_input += self.alpha1*(self.pe[:text.size(1)].unsqueeze(1))

              ### Speech Synthesis ###
              hidden_states = encoder_input
              text_mask = text.new_zeros(1,text.size(1)).to(torch.bool)
              for layer in self.Encoder:
                  hidden_states, _ = layer(hidden_states,
                                          src_key_padding_mask=text_mask)


              #with info about mel
               
             
              mel_info = self.mylin3(melstext)
             

              ### Duration Predictor ###

              durations = self.Duration(hidden_states.permute(1,2,0))

              if (durations.size(1)- mel_info.size(1))>=0 :
                   mel_info = torch.nn.functional.pad(mel_info  ,  (1,durations.size(1)- mel_info.size(1)-1 ), "constant", 0)
              else:
                  durations = torch.nn.functional.pad(durations ,  (1,mel_info.size(1)-durations.size(1) -1), "constant", 0)

              durations += mel_info 



              hidden_states_expanded = self.LR(hidden_states, durations, alpha, inference=True)

              pitch = self.Pitch(hidden_states_expanded.permute(1,2,0))
              energy = self.Energy(hidden_states_expanded.permute(1,2,0))

              pitch_one_hot = pitch_to_one_hot(pitch, False)
              energy_one_hot = energy_to_one_hot(energy, False)

              
              hidden_states_expanded = hidden_states_expanded + pitch_one_hot.transpose(1,0) + energy_one_hot.transpose(1,0)       #check for all device attributes
        
              hidden_states_expanded += self.alpha2*self.pe[:hidden_states_expanded.size(0)].unsqueeze(1)
              

              
              mel_mask = text.new_zeros(1, hidden_states_expanded.size(0)).to(torch.bool)

              for layer in self.Decoder:
                  hidden_states_expanded, _ = layer(hidden_states_expanded,
                                                    src_key_padding_mask=mel_mask)
              
              mel_out = self.Projection(hidden_states_expanded.transpose(0,1)).transpose(1,2)

              return (mel_out,pitch,energy,mel_out.shape[2])


def energy_to_one_hot(e, is_inference = False, is_log_output = False, offset = 1):                                        #check for scale
    # e = de_norm_mean_std(e, hp.e_mean, hp.e_std)
    # For pytorch > = 1.6.0
    bins = torch.linspace(hparams.e_min, hparams.e_max, steps=255).to(torch.device("cuda" if hparams.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        e = torch.clamp(torch.round(e.exp() - offset), min=0).long()
        
    e_quantize = bucketize(e.to(torch.device("cuda" if hparams.ngpu > 0 else "cpu")), bins)

    return F.one_hot(e_quantize.long(), 256).float()
    
    
def pitch_to_one_hot(f0, is_inference = False, is_log_output = False, offset = 1):
    # Required pytorch >= 1.6.0
    # f0 = de_norm_mean_std(f0, hp.f0_mean, hp.f0_std)
    bins = torch.exp(torch.linspace(np.log(hparams.p_min), np.log(hparams.p_max), 255)).to(torch.device("cuda" if hparams.ngpu > 0 else "cpu"))
    if is_inference and is_log_output:
        f0 = torch.clamp(torch.round(f0.exp() - offset), min=0).long()
        
    p_quantize = bucketize(f0.to(torch.device("cuda" if hparams.ngpu > 0 else "cpu")), bins)
    #p_quantize = p_quantize - 1  # -1 to convert 1 to 256 --> 0 to 255
    return F.one_hot(p_quantize.long(), 256).float()

def bucketize(tensor, bucket_boundaries):
    result = torch.zeros_like(tensor, dtype=torch.int64)
    for boundary in bucket_boundaries:
        result += (tensor > boundary).int()
    return result


model3 = MyFastSpeech()
    
    
for parameter in model3.parameters():
    parameter.requires_grad = False    
    
    
    
for parameter in model3.Embedding.parameters():
    parameter.requires_grad = True

for parameter in model3.mylin3.parameters():
    parameter.requires_grad = True      
    
#for parameter in model3.Duration.parameters():
 #   parameter.requires_grad = True     
    
#for parameter in model3.Pitch.parameters():
 #   parameter.requires_grad = True     
    
#for parameter in model3.Energy.parameters():
#    parameter.requires_grad = True       
    
for parameter in model3.Projection.parameters():
    parameter.requires_grad = True         



print(f'The model has {count_parameters(model3):,} trainable parameters')



model3.to(device)
