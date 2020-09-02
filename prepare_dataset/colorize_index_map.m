function color_map = colorize_index_map(color_theme, index_map, theme_nums)
    % color_theme: ��ɫ����, uint8����
    % idnex_map:  ��Ӧ��index map, uint8����
    % color_map:  ��ɫ���index map, uint8����
    
    [H, W] = size(index_map);
    color_map_r = uint8(zeros(H, W, 1));
    color_map_g = uint8(zeros(H, W, 1));
    color_map_b = uint8(zeros(H, W, 1));
    
%     theme_nums = max(index_map(:));
    for i = 1 : theme_nums
        color_map_r(index_map == i) = color_theme(1, i, 1);
        color_map_g(index_map == i) = color_theme(1, i, 2);
        color_map_b(index_map == i) = color_theme(1, i, 3);
    end
    color_map = cat(3, color_map_r, color_map_g, color_map_b);
    
end