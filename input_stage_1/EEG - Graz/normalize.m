function [out] = normalize(in)

out = [];
for channel = 1:size(in,2)
    stream = in(:,channel);
    stream = (stream-min(stream))/(max(stream)-min(stream));
    out = [out,stream];
end

