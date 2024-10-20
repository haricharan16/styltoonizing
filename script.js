document.getElementById('semanticImage').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('semanticPreview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('stylerImage').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('stylerPreview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('processButton').addEventListener('click', function() {
    const semanticImageSrc = document.getElementById('semanticPreview').src;
    const stylerImageSrc = document.getElementById('stylerPreview').src;

    if (!semanticImageSrc || !stylerImageSrc) {
        alert('Please upload both images');
        return;
    }

    // Simulate image processing for Cartoonized Image
    setTimeout(function() {
        document.getElementById('cartoonizedImage').src = semanticImageSrc;
    }, 1000);

    // Simulate image processing for Image Style Transfer
    setTimeout(function() {
        document.getElementById('styledImage').src = stylerImageSrc;
    }, 2000);
});
