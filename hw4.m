[y, fs] = audioread( "C:/Users/Mason/Downloads/HW4/audio.wav");
y = y(:,1); % just one channel
N = length( y );
nFFT = 1024; % This can be changed for better/worse resolution
hop = floor( nFFT / 4 );
nFrames = floor( N / hop ) - 1; % approximately
%printf("%d frames.\n", nFrames);
F = zeros( nFFT, nFrames );
w = hanning( nFFT ); % Window choice, again, this can be changed
for n = 1:nFrames
iStart = (n-1) * hop;
if iStart+nFFT > N, break; end
F(:, n) = fft( w .* y(iStart+1 : iStart+nFFT) );
G(:, n) = 20*log10(abs(F(1:nFFT/2, n)));
end

%read samples from .ascii file 
fid = fopen( "C:/Users/Mason/Downloads/HW4/stft_.out" );
samples = fscanf( fid, "%f" );
fclose( fid );
samples = reshape( samples, 512, 4306 );

%make a sublpot of abs(F) and abs(samples)
imagesc( G );
xlim([0 4000])
ylim([0 100]);
imagesc(samples)
xlim([0 4000])
ylim([0 100]);
