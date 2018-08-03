% -------------------------------------------------------------------------------------------------
function [cx, cy, w, h] = get_axis_aligned_BB(region)
%GETAXISALIGNEDBB computes axis-aligned bbox with same area as the rotated one (REGION)
% -------------------------------------------------------------------------------------------------
nv = numel(region);
assert(nv==8 || nv==4);
																	%ground truth file format.
																	%frameN: X1, Y1, X2, Y2, X3, Y3, X4, Y4
if nv==8
    cx = mean(region(1:2:end));										%mean of x : centre x.
    cy = mean(region(2:2:end));										%			 centre y.
    x1 = min(region(1:2:end));										%x　の中で一番左上角と近いポイント。
    x2 = max(region(1:2:end));
    y1 = min(region(2:2:end));
    y2 = max(region(2:2:end));
    A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));			%norm 为取向量模。此处A1为旋转后矩形面积。
    A2 = (x2 - x1) * (y2 - y1);														%此处A2为旋转矩形外接正则矩形面积。
    s = sqrt(A1/A2);																%面积比的平方根。
    w = s * (x2 - x1) + 1;															%这样 将外接矩形按比例缩放，保证和原来的ground truth面积相同。
    h = s * (y2 - y1) + 1;
else
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x+w/2;
    cy = y+h/2;
end
