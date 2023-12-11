import React, { useEffect } from "react";
import "./App.css";

function App() {
  const matLab = `image = imread("Tufaha.jpg");

    %Basic Use Case Question
    %Make Image negative and show all 8 planes
    neg = negativeImage(image);
    bitPlanes(neg); 
    
    
    %Make Image Negative
    function img = negativeImage(image)
        L = 2 ^ 8;
        img = (L - 1) - image;
    end
    
    %Get All 8 Planes Of Image DOES NOT RETURN
    function bitPlanes(image)
        for plane = 1:8
        B = bitget(image, plane);
        subplot(2, 4, plane);
        imshow(255 * B);
        title(['Bit plane ' num2str(plane)]);
        end
    end
    
    %Get Graysacle Version Of Image
    function img = grayScale(image)
        img = rgb2gray(image);
    end
    
    %Remove salt noise
    function img = removeSaltNoise(image)
        filterSize = [5, 5];  
        grayImage = grayScale(image);
        img = medfilt2(grayImage, filterSize);
    end
    
    %Remove pepper noise
    function img = removePepperNoise(image)
        filterSize = [5, 5];  
        grayImage = grayScale(image);
        img = medfilt2(grayImage, filterSize);
    end
    
    %Gls with background
    function img = glsWithBackground(image)
        T1 = 100;
        T2 = 200;
        grayImage = grayScale(image);
        inside_mask = (grayImage >= T1) & (grayImage <= T2);
        output_with_background = grayImage; 
        output_with_background(inside_mask) = 255;
        img = output_with_background;
    end
    
    %Gls without background
    function img = glsWithoutBackground(image)
        T1 = 100;
        T2 = 200;
        grayImage = grayScale(image);
        outside_mask = (grayImage < T1) | (grayImage > T2);
        output_without_background = zeros(size(grayImage), 'uint8');
        output_without_background(outside_mask) = 255;
        img = output_without_background;
    end
    
    %Histogram Equalization DOES NOT RETURN
    function histogramEqualization(image)
        subplot(2,2,1);
        imshow(image)
        subplot(2,2,2);
        imhist(image,64);
        afterImg = histeq(image);
        subplot(2,2,3);
        imshow(afterImg)
        subplot(2,2,4);
        imhist(afterImg,64);
    end
    
    %Erosion
    function img = erosion(image)
        se = strel('disk', 5);
        img = imerode(image, se);
    end
    
    %Open
    function img = open(image)
        se = strel('disk', 5);
        img = imopen(image, se);
    end
    
    %Close
    function img = close(image)
        se = strel('disk', 5);
        img = imclose(image, se);
    end
    
    
    %Dialate
    function img = dialate(image)
        se = strel('disk', 5);
        img = imdilate(image, se);
    end
    
    %Power Log
    function img = powerLog(image)
        input_image = im2double(image);
        gamma = 5
        img = input_image.^gamma;
    end
    
    %Edge Detection
    function img = edgeDetection(image)
        newImage = grayScale(image);
        img = edge(newImage,'Prewitt');
        %img = edge(newImage,'Canny');
    end
    
    %High Pass Butterworth Filter DOES NOT RETURN
    function highPassButterWorth(image)
        grayImage = rgb2gray(image);
        fftImage = fft2(double(grayImage));
        [M, N] = size(fftImage);
        u = 0:(M-1);
        v = 0:(N-1);
        u(u > M/2) = u(u > M/2) - M;
        v(v > N/2) = v(v > N/2) - N;
        [V, U] = meshgrid(v, u);
        D0 = 30; 
        n = 2;   
        H_gaussian_highpass = 1 - exp(-(U.^2 + V.^2) / (2 * D0^2));
        H_butterworth_highpass = 1 ./ (1 + (D0 ./ sqrt(U.^2 + V.^2)).^(2 * n));
        filteredImage_gaussian_highpass = real(ifft2(fftImage .* H_gaussian_highpass));
        filteredImage_butterworth_highpass = real(ifft2(fftImage .* H_butterworth_highpass));
        subplot(2, 2, 1);
        imshow(grayImage);
        title('Original Image');
        subplot(2, 2, 2);
        imshow(filteredImage_gaussian_highpass, []);
        title('Gaussian High Pass Filtered Image');
        subplot(2, 2, 3);
        imshow(filteredImage_butterworth_highpass, []);
        title('Butterworth High Pass Filtered Image');
    end
    
    %Low Pass Filter DOES NOT RETURN
    function lowPassFilter(image)
    grayImage = rgb2gray(image);
    fftImage = fft2(double(grayImage));
    [M, N] = size(fftImage);
    u = 0:(M-1);
    v = 0:(N-1);
    u(u > M/2) = u(u > M/2) - M;
    v(v > N/2) = v(v > N/2) - N;
    [V, U] = meshgrid(v, u);
    D0 = 30; 
    n = 2;   
    H_ideal = double(sqrt(U.^2 + V.^2) <= D0);
    H_gaussian = exp(-(U.^2 + V.^2) / (2 * D0^2));
    H_butterworth = 1 ./ (1 + (sqrt(U.^2 + V.^2) / D0).^(2 * n));
    filteredImage_ideal = real(ifft2(fftImage .* H_ideal));
    filteredImage_gaussian = real(ifft2(fftImage .* H_gaussian));
    filteredImage_butterworth = real(ifft2(fftImage .* H_butterworth));
    subplot(2, 2, 1);
    imshow(grayImage);
    title('Original Image');
    subplot(2, 2, 2);
    imshow(filteredImage_ideal, []);
    title('Ideal Low Pass Filtered Image');
    subplot(2, 2, 3);
    imshow(filteredImage_gaussian, []);
    title('Gaussian Low Pass Filtered Image');
    subplot(2, 2, 4);
    imshow(filteredImage_butterworth, []);
    title('Butterworth Low Pass Filtered Image');
    end
    `;

  useEffect(() => {
    copyCodeToClipboard(matLab);
  }, []);
  const copyCodeToClipboard = (code) => {
    navigator.clipboard
      .writeText(code)
      .then(() => {
        console.log("Code copied to clipboard");
      })
      .catch((err) => {
        console.error("Unable to copy code to clipboard", err);
      });
  };

  return (
    <div className="App">
      <button onClick={() => copyCodeToClipboard(matLab)}>MAT LAB</button>
    </div>
  );
}

export default App;
{
  /* <button onClick={() => copyCodeToClipboard(graphColoring)}>
        Graph Coloring
      </button>
      <button onClick={() => copyCodeToClipboard(xor)}>Xor </button>
      <button onClick={() => copyCodeToClipboard(kanpsack)}>Knapsack </button>
      <button onClick={() => copyCodeToClipboard(dfs)}>DFS </button>
      <button onClick={() => copyCodeToClipboard(water)}>Water</button>
      <button onClick={() => copyCodeToClipboard(token)}>Token</button>
      <button onClick={() => copyCodeToClipboard(matLab)}>MAT LAB</button>
      <button onClick={() => copyCodeToClipboard(matLab)}>MAT LAB</button> */
}
